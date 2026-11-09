// Tabs: projection onto a line, eigenfaces. Each tab keeps its own state
// while hidden; the hash names the open tab.

const HASH = { proj: "#projection", faces: "#eigenfaces" };
const SHOW = { proj: projShow, faces: facesShow };

function setMode(mode) {
  history.replaceState(null, "", HASH[mode]);
  for (const t of document.querySelectorAll(".tab")) {
    t.setAttribute("aria-selected", t.dataset.mode === mode);
  }
  for (const e of document.querySelectorAll("[data-show]")) {
    e.hidden = !e.dataset.show.split(" ").includes(mode);
  }
  SHOW[mode]();
}

for (const t of document.querySelectorAll(".tab")) {
  t.onclick = () => setMode(t.dataset.mode);
}

setMode(Object.keys(HASH).find(m => HASH[m] === location.hash) || "proj");
