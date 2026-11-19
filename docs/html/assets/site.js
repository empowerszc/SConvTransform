(function () {
  "use strict";
  document.documentElement.classList.add("js");

  // ---------- theme ----------
  var stored = null;
  try { stored = localStorage.getItem("sconv-theme"); } catch (e) {}
  var theme = stored
    || (window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
  function applyTheme(t) {
    document.documentElement.setAttribute("data-theme", t);
  }
  applyTheme(theme);
  var btn = document.querySelector(".theme-btn");
  if (btn) {
    btn.addEventListener("click", function () {
      theme = (document.documentElement.getAttribute("data-theme") === "dark") ? "light" : "dark";
      applyTheme(theme);
      try { localStorage.setItem("sconv-theme", theme); } catch (e) {}
    });
  }

  // ---------- active nav item ----------
  var page = (location.pathname.split("/").pop() || "index.html");
  document.querySelectorAll(".nav a").forEach(function (a) {
    var href = a.getAttribute("href") || "";
    if (href === page) a.classList.add("active");
  });

  // ---------- reading progress ----------
  var bar = document.createElement("div");
  bar.className = "progress";
  document.body.appendChild(bar);
  function onScroll() {
    var h = document.documentElement;
    var max = h.scrollHeight - h.clientHeight;
    bar.style.width = (max > 0 ? (h.scrollTop / max) * 100 : 0) + "%";
    if (toTop) toTop.classList.toggle("show", h.scrollTop > 600);
  }
  window.addEventListener("scroll", onScroll, { passive: true });

  // ---------- back-to-top ----------
  var toTop = document.createElement("button");
  toTop.className = "to-top";
  toTop.type = "button";
  toTop.setAttribute("aria-label", "回到顶部");
  toTop.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M12 19V5M5 12l7-7 7 7"/></svg>';
  toTop.addEventListener("click", function () {
    window.scrollTo({ top: 0, behavior: "smooth" });
  });
  document.body.appendChild(toTop);
  onScroll();

  // ---------- reveal-on-scroll ----------
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
