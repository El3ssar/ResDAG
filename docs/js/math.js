// KaTeX rendering for pymdownx.arithmatex (generic mode).
document$.subscribe(() => {
  const blocks = document.querySelectorAll(".arithmatex");
  if (!blocks.length || typeof renderMathInElement !== "function") return;
  blocks.forEach((el) => {
    renderMathInElement(el, {
      delimiters: [
        { left: "\\(", right: "\\)", display: false },
        { left: "\\[", right: "\\]", display: true },
        { left: "$$", right: "$$", display: true },
        { left: "$", right: "$", display: false },
      ],
      throwOnError: false,
    });
  });
});
