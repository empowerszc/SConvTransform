(function () {
  document.documentElement.classList.add("js");

  // Active nav item (works for file:// and http://).
  var page = (location.pathname.split("/").pop() || "index.html");
  document.querySelectorAll(".nav a").forEach(function (a) {
    var href = a.getAttribute("href") || "";
    if (href === page) a.classList.add("active");
  });

  // Reading progress bar.
  var bar = document.createElement("div");
  bar.className = "progress";
  document.body.appendChild(bar);
  function onScroll() {
    var h = document.documentElement;
    var max = h.scrollHeight - h.clientHeight;
    bar.style.width = (max > 0 ? (h.scrollTop / max) * 100 : 0) + "%";
  }
  window.addEventListener("scroll", onScroll, { passive: true });
  onScroll();

  // Subtle reveal-on-scroll for top-level blocks.
  var targets = document.querySelectorAll(".section > .wrap > *, .metric-strip > *, .timeline > *");
  if ("IntersectionObserver" in window) {
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (e) {
        if (e.isIntersecting) {
          e.target.classList.add("in");
          io.unobserve(e.target);
        }
      });
    }, { rootMargin: "0px 0px -8% 0px", threshold: 0.04 });
    targets.forEach(function (el, i) {
      el.classList.add("reveal");
      el.style.transitionDelay = Math.min(i % 6, 5) * 40 + "ms";
      io.observe(el);
    });
  }
})();
