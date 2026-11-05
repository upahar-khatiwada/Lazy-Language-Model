// Text from data files, with names written in backticks shown as code. No markup is interpreted.
export function InlineCode({ text }: { text: string }) {
  return (
    <>
      {text
        .split("`")
        .map((part, index) => (index % 2 === 1 ? <code key={index}>{part}</code> : part))}
    </>
  );
}
