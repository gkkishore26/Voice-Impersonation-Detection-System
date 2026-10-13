/* ============ VIDS motion.js — Hermes motion runtime (additive; predict flow untouched) ============ */
(function () {
  'use strict';
  var reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  /* ---------- Lenis smooth scroll (guarded; Hermes uses html.lenis) ---------- */
  try {
    if (!reduce && typeof window.Lenis !== 'undefined') {
      var lenis = new window.Lenis({ anchors: true, lerp: 0.12 });
      var raf = function (t) { lenis.raf(t); requestAnimationFrame(raf); };
      requestAnimationFrame(raf);
    }
  } catch (e) { /* native smooth scroll fallback (CSS) */ }

  /* ---------- reveal on scroll (claim-rise + stagger via --d) ---------- */
  var revealEls = document.querySelectorAll('.rv');
  if ('IntersectionObserver' in window && !reduce) {
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (en) {
        if (en.isIntersecting) { en.target.classList.add('in'); io.unobserve(en.target); }
      });
    }, { threshold: 0.12, rootMargin: '0px 0px -6% 0px' });
    revealEls.forEach(function (el) { io.observe(el); });
  } else {
    revealEls.forEach(function (el) { el.classList.add('in'); });
  }

  /* ---------- parallax drift (mirrors --py-img translate formula) ---------- */
  var plxEls = Array.prototype.slice.call(document.querySelectorAll('[data-plx]'));
  if (plxEls.length && !reduce) {
    var ticking = false;
    var update = function () {
      ticking = false;
      var vh = window.innerHeight || 800;
      plxEls.forEach(function (el) {
        var r = el.getBoundingClientRect();
        var dir = parseFloat(el.getAttribute('data-plx')) || -1;
        var progress = (r.top + r.height / 2 - vh / 2) / vh; // -0.5..0.5-ish
        var art = el.matches('svg,img') ? el : el.querySelector('svg, img');
        if (art) art.style.setProperty('--py-img', (progress * 64 * dir).toFixed(1) + 'px');
      });
    };
    window.addEventListener('scroll', function () {
      if (!ticking) { ticking = true; requestAnimationFrame(update); }
    }, { passive: true });
    update();
  }

  /* ---------- count-up spec numbers ---------- */
  var counters = document.querySelectorAll('[data-count]');
  var runCounter = function (el) {
    if (el.dataset.done) return;
    el.dataset.done = '1';
    var target = parseFloat(el.dataset.count), suffix = el.dataset.suffix || '';
    var dec = (el.dataset.count.split('.')[1] || '').length;
    if (reduce) { el.textContent = target.toFixed(dec) + suffix; return; }
    el.classList.add('counting');
    var t0 = performance.now(), dur = 1100;
    var step = function (t) {
      var p = Math.min(1, (t - t0) / dur), e = 1 - Math.pow(1 - p, 4);
      el.textContent = (target * e).toFixed(dec) + suffix;
      if (p < 1) requestAnimationFrame(step);
      else el.classList.remove('counting');
    };
    requestAnimationFrame(step);
  };
  if ('IntersectionObserver' in window && !reduce) {
    var cio = new IntersectionObserver(function (entries) {
      entries.forEach(function (en) { if (en.isIntersecting) { runCounter(en.target); cio.unobserve(en.target); } });
    }, { threshold: 0.4 });
    counters.forEach(function (el) { cio.observe(el); });
  }

  /* ---------- spec terminal tabs + copy ---------- */
  var tabs = document.querySelectorAll('.term-tab');
  tabs.forEach(function (tab) {
    tab.addEventListener('click', function () {
      tabs.forEach(function (t) { t.classList.toggle('active', t === tab); });
      document.querySelectorAll('.term-panel').forEach(function (p) {
        p.classList.toggle('active', p.dataset.panel === tab.dataset.term);
      });
    });
  });
  var copyBtn = document.getElementById('termCopy');
  if (copyBtn) {
    copyBtn.addEventListener('click', function () {
      var active = document.querySelector('.term-panel.active');
      var txt = active ? active.innerText : '';
      var done = function () { copyBtn.textContent = 'Copied'; setTimeout(function () { copyBtn.textContent = 'Copy'; }, 1200); };
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(txt).then(done).catch(done);
      } else { done(); }
    });
  }

  /* ---------- verdict attention pulse (observes result card, zero-touch) ---------- */
  var verdict = document.getElementById('verdict');
  var resultCard = document.getElementById('resultCard');
  if (verdict && resultCard && 'MutationObserver' in window && !reduce) {
    new MutationObserver(function () {
      if (!resultCard.hidden) {
        verdict.classList.remove('attn');
        void verdict.offsetWidth; // restart animation
        verdict.classList.add('attn');
      }
    }).observe(resultCard, { attributes: true, attributeFilter: ['hidden'] });
  }
})();
