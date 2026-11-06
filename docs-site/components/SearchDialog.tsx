"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useEffect, useEffectEvent, useRef, useState } from "react";
import { revealHref } from "@/lib/reveal";
import {
  highlightParts,
  nextActive,
  NO_RESULTS,
  opensSearch,
  SEARCH_INDEX_FILE,
  type SearchHit,
  type SearchResults,
} from "@/lib/search";
import { Button } from "./Button";
import styles from "./SearchDialog.module.css";

const BASE_PATH = process.env.NEXT_PUBLIC_BASE_PATH ?? "";
const LISTBOX_ID = "search-results";
const optionId = (index: number) => `search-option-${index}`;

type Status = "idle" | "loading" | "ready" | "failed";
type Engine = (query: string) => SearchResults;

function Highlighted({ text, terms }: { text: string; terms: string[] }) {
  return (
    <>
      {highlightParts(text, terms).map((part, index) =>
        part.match ? <mark key={index}>{part.text}</mark> : part.text,
      )}
    </>
  );
}

export function SearchDialog() {
  const router = useRouter();
  const dialogRef = useRef<HTMLDialogElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const engineRef = useRef<Engine | null>(null);
  const queryRef = useRef("");
  const [status, setStatus] = useState<Status>("idle");
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<SearchResults>(NO_RESULTS);
  const [active, setActive] = useState(0);

  // Runs on first open only, so a page that never searches loads neither MiniSearch nor the index.
  async function loadEngine() {
    setStatus("loading");
    try {
      const [{ loadIndex, runSearch }, response] = await Promise.all([
        import("@/lib/search-engine"),
        fetch(`${BASE_PATH}/${SEARCH_INDEX_FILE}`),
      ]);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const index = loadIndex(await response.text());
      engineRef.current = (text) => runSearch(index, text);
      setResults(engineRef.current(queryRef.current));
      setActive(0);
      setStatus("ready");
    } catch {
      setStatus("failed");
    }
  }

  function open() {
    const dialog = dialogRef.current;
    if (!dialog) return;
    if (dialog.open) {
      inputRef.current?.select();
      return;
    }
    dialog.showModal();
    inputRef.current?.focus();
    if (status === "idle" || status === "failed") void loadEngine();
  }

  function close() {
    dialogRef.current?.close();
  }

  function reset() {
    queryRef.current = "";
    setQuery("");
    setResults(NO_RESULTS);
    setActive(0);
  }

  function search(text: string) {
    queryRef.current = text;
    setQuery(text);
    setResults(engineRef.current ? engineRef.current(text) : NO_RESULTS);
    setActive(0);
  }

  function onInputKey(event: React.KeyboardEvent<HTMLInputElement>) {
    if (event.key === "ArrowDown" || event.key === "ArrowUp") {
      event.preventDefault();
      setActive(nextActive(active, event.key === "ArrowDown" ? 1 : -1, results.flat.length));
    } else if (event.key === "Enter") {
      const hit = results.flat[active];
      if (!hit) return;
      event.preventDefault();
      close();
      // The result may be folded on this page. On another page, SectionReveal opens it on arrival.
      revealHref(hit.href);
      router.push(hit.href);
    }
  }

  const onWindowKey = useEffectEvent((event: KeyboardEvent) => {
    const target = event.target as HTMLElement | null;
    const info = {
      key: event.key,
      ctrlKey: event.ctrlKey,
      metaKey: event.metaKey,
      altKey: event.altKey,
      tagName: target?.tagName,
      isContentEditable: target?.isContentEditable,
    };
    if (!opensSearch(info)) return;
    event.preventDefault();
    open();
  });

  useEffect(() => {
    const listener = (event: KeyboardEvent) => onWindowKey(event);
    window.addEventListener("keydown", listener);
    return () => window.removeEventListener("keydown", listener);
  }, []);

  // Keeps the option chosen with the arrow keys in view.
  useEffect(() => {
    document.getElementById(optionId(active))?.scrollIntoView({ block: "nearest" });
  }, [active, results]);

  const typed = query.trim().length > 0;
  let message = "";
  if (status === "failed") message = "The search index could not be loaded.";
  else if (status !== "ready") message = "Loading the search index ...";
  else if (typed && results.total === 0) message = `No results for "${query.trim()}".`;
  else if (typed) message = `${results.total} ${results.total === 1 ? "result" : "results"}.`;

  let position = 0;
  const option = (hit: SearchHit, label: string) => {
    const index = position++;
    return (
      <Link
        prefetch={false}
        key={hit.id}
        id={optionId(index)}
        href={hit.href}
        role="option"
        aria-selected={index === active}
        tabIndex={-1}
        className={`${styles.option} ${index === active ? styles.active : ""}`}
        onClick={close}
        onMouseMove={() => setActive(index)}
      >
        <span className={styles.badge}>{label}</span>
        <span className={styles.text}>
          <span className={styles.title}>
            <Highlighted text={hit.title} terms={hit.terms} />
          </span>
          {hit.detail ? (
            <span className={styles.detail}>
              <Highlighted text={hit.detail} terms={hit.terms} />
            </span>
          ) : null}
          {hit.formula ? (
            <code className={styles.formula}>
              <Highlighted text={hit.formula} terms={hit.terms} />
            </code>
          ) : null}
        </span>
      </Link>
    );
  };

  return (
    <>
      <Button
        onClick={open}
        aria-haspopup="dialog"
        aria-keyshortcuts="Control+K Meta+K /"
        title="Search the documentation (Ctrl+K, Cmd+K or /)"
      >
        Search <kbd className={styles.key}>/</kbd>
      </Button>
      <dialog
        ref={dialogRef}
        className={styles.dialog}
        aria-label="Search the documentation"
        onClose={reset}
        onClick={(event) => {
          if (event.target === dialogRef.current) close();
        }}
      >
        <div className={styles.panel}>
          <input
            ref={inputRef}
            className={styles.input}
            type="text"
            role="combobox"
            aria-label="Search concepts, equations, pages, files and functions"
            aria-expanded={results.flat.length > 0}
            aria-controls={LISTBOX_ID}
            aria-activedescendant={results.flat.length > 0 ? optionId(active) : undefined}
            aria-autocomplete="list"
            autoComplete="off"
            spellCheck={false}
            placeholder="Search concepts, equations, pages, files, functions"
            value={query}
            onChange={(event) => search(event.target.value)}
            onKeyDown={onInputKey}
          />
          <p className={styles.status} role="status">
            {message}
          </p>
          <div id={LISTBOX_ID} className={styles.results} role="listbox" aria-label="Results">
            {results.groups.map((group) => (
              <div key={group.type} role="group" aria-label={group.label}>
                <div className={styles.groupTitle} aria-hidden="true">
                  {group.label}
                  {group.total > group.hits.length
                    ? ` (top ${group.hits.length} of ${group.total})`
                    : ""}
                </div>
                {group.hits.map((hit) => option(hit, group.label))}
              </div>
            ))}
          </div>
          <p className={styles.hint}>
            <kbd className={styles.key}>↑</kbd> <kbd className={styles.key}>↓</kbd> move,{" "}
            <kbd className={styles.key}>Enter</kbd> open, <kbd className={styles.key}>Esc</kbd>{" "}
            close
          </p>
        </div>
      </dialog>
    </>
  );
}
