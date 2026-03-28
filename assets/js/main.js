document.addEventListener("DOMContentLoaded", () => {
    const yearEl = document.getElementById("year");
    if (yearEl) {
      yearEl.textContent = new Date().getFullYear().toString();
    }
  
    const copyBtn = document.getElementById("copyBibtexBtn");
    const bibtexBlock = document.getElementById("bibtexBlock");
  
    if (copyBtn && bibtexBlock) {
      copyBtn.addEventListener("click", async () => {
        const text = bibtexBlock.innerText.trim();
  
        try {
          await navigator.clipboard.writeText(text);
          const oldText = copyBtn.textContent;
          copyBtn.textContent = "Copied";
          setTimeout(() => {
            copyBtn.textContent = oldText;
          }, 1400);
        } catch (error) {
          copyBtn.textContent = "Copy failed";
          setTimeout(() => {
            copyBtn.textContent = "Copy BibTeX";
          }, 1400);
        }
      });
    }
  });