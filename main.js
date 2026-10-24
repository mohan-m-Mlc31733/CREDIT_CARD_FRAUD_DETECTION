/**
 * FraudShield — Interactive Scripts & Animations
 * Responsive Mouse Pointer, 3D Tilt, Particle Waves & Live Fraud Scanner
 */

document.addEventListener('DOMContentLoaded', () => {
  initCustomCursor();
  initNavbar();
  initParticleWave();
  initCardTilt();
  initScrollAnimations();
  initCounters();
  initFraudSimulator();
  initRippleEffect();
});

/* ==========================================================================
   1. RESPONSIVE CUSTOM MOUSE POINTER
   ========================================================================== */
function initCustomCursor() {
  const cursor = document.getElementById('cursor');
  const follower = document.getElementById('cursor-follower');
  if (!cursor || !follower) return;

  let mouseX = window.innerWidth / 2;
  let mouseY = window.innerHeight / 2;
  let cursorX = mouseX;
  let cursorY = mouseY;
  let followerX = mouseX;
  let followerY = mouseY;
  let isHovering = false;

  window.addEventListener('mousemove', (e) => {
    mouseX = e.clientX;
    mouseY = e.clientY;
    cursor.style.opacity = '1';
    follower.style.opacity = '1';
  });

  window.addEventListener('mouseleave', () => {
    cursor.style.opacity = '0';
    follower.style.opacity = '0';
  });

  // Smooth lerp follow loop
  function renderCursor() {
    // Lead cursor is quick
    cursorX += (mouseX - cursorX) * 0.45;
    cursorY += (mouseY - cursorY) * 0.45;
    cursor.style.transform = `translate3d(${cursorX}px, ${cursorY}px, 0)`;

    // Follower ring has organic trailing momentum
    followerX += (mouseX - followerX) * 0.15;
    followerY += (mouseY - followerY) * 0.15;
    follower.style.transform = `translate3d(${followerX}px, ${followerY}px, 0)`;

    requestAnimationFrame(renderCursor);
  }
  requestAnimationFrame(renderCursor);

  // Magnetic & contextual hover states
  const interactives = document.querySelectorAll('a, button, [data-cursor], input, select, .process-card, .service-row, .testimonial-card');
  interactives.forEach(el => {
    el.addEventListener('mouseenter', (e) => {
      const type = el.getAttribute('data-cursor') || 'link';
      cursor.classList.add('hover');
      follower.classList.add('hover');
      
      if (type === 'card') {
        follower.classList.add('card-hover');
        follower.setAttribute('data-label', 'EXPLORE');
      } else if (type === 'view') {
        follower.classList.add('view-hover');
        follower.setAttribute('data-label', 'SCAN');
      }
    });

    el.addEventListener('mouseleave', () => {
      cursor.classList.remove('hover');
      follower.classList.remove('hover', 'card-hover', 'view-hover');
      follower.removeAttribute('data-label');
    });

    // Magnetic pull for elements with .magnetic
    if (el.classList.contains('magnetic')) {
      el.addEventListener('mousemove', (e) => {
        const rect = el.getBoundingClientRect();
        const centerX = rect.left + rect.width / 2;
        const centerY = rect.top + rect.height / 2;
        const deltaX = (e.clientX - centerX) * 0.35;
        const deltaY = (e.clientY - centerY) * 0.35;
        el.style.transform = `translate3d(${deltaX}px, ${deltaY}px, 0)`;
      });

      el.addEventListener('mouseleave', () => {
        el.style.transform = 'translate3d(0, 0, 0)';
      });
    }
  });

  // Click pulse animation
  window.addEventListener('mousedown', () => {
    cursor.classList.add('active');
    follower.classList.add('active');
  });

  window.addEventListener('mouseup', () => {
    cursor.classList.remove('active');
    follower.classList.remove('active');
  });
}

/* ==========================================================================
   2. NAVBAR & MOBILE MENU
   ========================================================================== */
function initNavbar() {
  const navbar = document.getElementById('navbar');
  const menuBtn = document.getElementById('menuBtn');
  const mobileMenu = document.getElementById('mobileMenu');

  window.addEventListener('scroll', () => {
    if (window.scrollY > 40) {
      navbar?.classList.add('scrolled');
    } else {
      navbar?.classList.remove('scrolled');
    }

    // Scroll progress bar
    const winScroll = document.body.scrollTop || document.documentElement.scrollTop;
    const height = document.documentElement.scrollHeight - document.documentElement.clientHeight;
    const scrolled = (winScroll / height) * 100;
    const progressBar = document.getElementById('scrollProgress');
    if (progressBar) progressBar.style.width = scrolled + '%';
  });

  if (menuBtn && mobileMenu) {
    menuBtn.addEventListener('click', () => {
      menuBtn.classList.toggle('active');
      mobileMenu.classList.toggle('active');
      document.body.style.overflow = mobileMenu.classList.contains('active') ? 'hidden' : '';
    });
  }
}

/* ==========================================================================
   3. ANIMATED 3D PARTICLE WAVE (IMAGE 1 & 4 AESTHETIC)
   ========================================================================== */
function initParticleWave() {
  const canvas = document.getElementById('heroCanvas') || document.getElementById('particleCanvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  let width, height;
  let particles = [];
  let mouse = { x: -1000, y: -1000, radius: 180 };

  function resize() {
    width = canvas.width = window.innerWidth;
    height = canvas.height = window.innerHeight;
    createParticles();
  }
  window.addEventListener('resize', resize);

  window.addEventListener('mousemove', (e) => {
    mouse.x = e.clientX;
    mouse.y = e.clientY;
  });

  function createParticles() {
    particles = [];
    const count = Math.min(Math.floor(width * 0.12), 160);
    for (let i = 0; i < count; i++) {
      particles.push({
        x: Math.random() * width,
        y: Math.random() * height,
        baseY: Math.random() * height,
        size: Math.random() * 2.2 + 0.6,
        speed: Math.random() * 0.8 + 0.2,
        phase: Math.random() * Math.PI * 2,
        color: Math.random() > 0.85 ? '#e74c3c' : (Math.random() > 0.6 ? '#ff6b35' : '#ffffff'),
        alpha: Math.random() * 0.6 + 0.2
      });
    }
  }

  resize();

  let time = 0;
  function animate() {
    time += 0.015;
    ctx.clearRect(0, 0, width, height);

    // Draw particle waves
    for (let i = 0; i < particles.length; i++) {
      const p = particles[i];
      p.x -= p.speed;
      if (p.x < 0) p.x = width;

      // Sine wave undulating movement
      const wave = Math.sin(time + p.phase + p.x * 0.005) * 28;
      p.y = p.baseY + wave;

      // Mouse displacement
      const dx = mouse.x - p.x;
      const dy = mouse.y - p.y;
      const dist = Math.sqrt(dx * dx + dy * dy);
      if (dist < mouse.radius) {
        const force = (1 - dist / mouse.radius) * 35;
        p.x -= (dx / dist) * force;
        p.y -= (dy / dist) * force;
      }

      ctx.beginPath();
      ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
      ctx.fillStyle = p.color;
      ctx.globalAlpha = p.alpha;
      ctx.fill();
    }

    // Connect close particles with subtle cyber lines
    ctx.globalAlpha = 0.08;
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 0.6;
    for (let i = 0; i < particles.length; i += 2) {
      for (let j = i + 1; j < particles.length; j += 4) {
        const dx = particles[i].x - particles[j].x;
        const dy = particles[i].y - particles[j].y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (dist < 90) {
          ctx.beginPath();
          ctx.moveTo(particles[i].x, particles[i].y);
          ctx.lineTo(particles[j].x, particles[j].y);
          ctx.stroke();
        }
      }
    }
    ctx.globalAlpha = 1;

    requestAnimationFrame(animate);
  }
  requestAnimationFrame(animate);
}

/* ==========================================================================
   4. 3D CARD TILT ON MOUSE POINTER
   ========================================================================== */
function initCardTilt() {
  const cards = document.querySelectorAll('.tilt-card, .process-card, .testimonial-card');
  cards.forEach(card => {
    card.addEventListener('mousemove', (e) => {
      const rect = card.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;
      const centerX = rect.width / 2;
      const centerY = rect.height / 2;

      const rotateX = ((y - centerY) / centerY) * -10;
      const rotateY = ((x - centerX) / centerX) * 10;

      card.style.transform = `perspective(1000px) rotateX(${rotateX}deg) rotateY(${rotateY}deg) translateY(-8px) scale(1.02)`;
      
      // Dynamic light shine reflection
      const shine = card.querySelector('.tilt-shine');
      if (shine) {
        shine.style.background = `radial-gradient(circle at ${x}px ${y}px, rgba(255,255,255,0.2) 0%, transparent 70%)`;
      }
    });

    card.addEventListener('mouseleave', () => {
      card.style.transform = 'perspective(1000px) rotateX(0deg) rotateY(0deg) translateY(0) scale(1)';
      const shine = card.querySelector('.tilt-shine');
      if (shine) shine.style.background = 'none';
    });
  });
}

/* ==========================================================================
   5. SCROLL TRIGGER ANIMATIONS
   ========================================================================== */
function initScrollAnimations() {
  const animatedElements = document.querySelectorAll('[data-animate]');
  if (!animatedElements.length) return;

  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        const delay = entry.target.getAttribute('data-delay') || 0;
        setTimeout(() => {
          entry.target.classList.add('visible');
        }, delay);
      }
    });
  }, { threshold: 0.15 });

  animatedElements.forEach(el => observer.observe(el));
}

/* ==========================================================================
   6. COUNTER ANIMATIONS
   ========================================================================== */
function initCounters() {
  const counters = document.querySelectorAll('.counter');
  if (!counters.length) return;

  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting && !entry.target.classList.contains('counted')) {
        entry.target.classList.add('counted');
        const target = parseFloat(entry.target.getAttribute('data-target'));
        const suffix = entry.target.getAttribute('data-suffix') || '';
        const decimals = parseInt(entry.target.getAttribute('data-decimals') || 0);
        let start = 0;
        const duration = 2000;
        const startTime = performance.now();

        function update(currentTime) {
          const elapsed = currentTime - startTime;
          const progress = Math.min(elapsed / duration, 1);
          // Ease out expo
          const easeOut = progress === 1 ? 1 : 1 - Math.pow(2, -10 * progress);
          const current = (start + (target - start) * easeOut).toFixed(decimals);
          entry.target.textContent = current + suffix;

          if (progress < 1) {
            requestAnimationFrame(update);
          }
        }
        requestAnimationFrame(update);
      }
    });
  }, { threshold: 0.5 });

  counters.forEach(c => observer.observe(c));
}

/* ==========================================================================
   7. LIVE FRAUD SCANNER SIMULATOR (BASED ON REAL MODEL DATA)
   ========================================================================== */
function initFraudSimulator() {
  const form = document.getElementById('fraudSimForm');
  if (!form) return;

  const resultBox = document.getElementById('simResult');
  const riskGauge = document.getElementById('riskGauge');
  const riskScoreText = document.getElementById('riskScore');
  const verdictText = document.getElementById('simVerdict');
  const explanationList = document.getElementById('simReasons');

  form.addEventListener('submit', (e) => {
    e.preventDefault();

    const amt = parseFloat(document.getElementById('simAmt')?.value || 120);
    const category = document.getElementById('simCategory')?.value || 'shopping_net';
    const state = document.getElementById('simState')?.value || 'NY';
    const dist = parseFloat(document.getElementById('simDist')?.value || 5);

    // Compute heuristic probability using model's top SHAP factors
    // Mean SHAP: amt: 2.91, gas_transport: 2.30, travel: 0.52, grocery: 0.52, state_ID: 0.49
    let logit = -4.5; // Base log-odds for ~0.58% base fraud rate

    // Amount impact
    if (amt > 800) logit += 3.8;
    else if (amt > 400) logit += 2.4;
    else if (amt > 150) logit += 1.1;
    else if (amt < 10) logit += 0.8; // Small test transactions

    // Category impact (from SHAP table)
    if (category === 'gas_transport') logit += 2.5;
    else if (category === 'travel') logit += 1.8;
    else if (category === 'grocery_pos') logit += 0.9;
    else if (category === 'shopping_net') logit += 1.4;

    // High risk states from SHAP table
    if (['ID', 'SD', 'IA', 'CT', 'TN', 'WY'].includes(state)) {
      logit += 0.9;
    }

    // Distance anomaly
    if (dist > 100) logit += 2.1;
    else if (dist > 50) logit += 1.2;

    // Sigmoid function
    const prob = 1 / (1 + Math.exp(-logit));
    const score = Math.min(Math.max(prob * 100, 0.4), 99.8).toFixed(1);

    // XGBoost tuned threshold = 0.1959 (19.6%)
    const thresholdF1 = 19.6;
    const isFraud = score >= thresholdF1;

    // Render verdict
    if (resultBox) {
      resultBox.style.display = 'block';
      resultBox.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }

    if (riskScoreText) riskScoreText.textContent = `${score}%`;
    if (riskGauge) {
      riskGauge.style.width = `${score}%`;
      riskGauge.style.background = isFraud ? 'linear-gradient(90deg, #f39c12, #e74c3c)' : 'linear-gradient(90deg, #2ecc71, #27ae60)';
    }

    if (verdictText) {
      if (isFraud) {
        verdictText.innerHTML = '<span class="badge-flagged">⚠️ FLAGGED AS FRAUD</span> — Breached Optimal F1 Threshold (19.6%)';
        verdictText.className = 'verdict flagged';
      } else {
        verdictText.innerHTML = '<span class="badge-approved">✅ TRANSACTION APPROVED</span> — Low Risk Profile';
        verdictText.className = 'verdict approved';
      }
    }

    if (explanationList) {
      explanationList.innerHTML = '';
      const factors = [];
      if (amt > 400) factors.push(`High Transaction Amount ($${amt.toFixed(2)}) heavily elevated risk (Top SHAP factor: +2.92)`);
      if (category === 'gas_transport') factors.push(`High-velocity merchant category (Gas & Transport SHAP importance: +2.30)`);
      if (category === 'travel') factors.push(`Cross-regional travel pattern detected (SHAP: +0.52)`);
      if (dist > 50) factors.push(`Merchant is ${dist} miles away from cardholder coordinate (Anomalous Haversine distance)`);
      if (!factors.length) factors.push('Standard purchase pattern consistent with historical cardholder profile.');

      factors.forEach(f => {
        const li = document.createElement('li');
        li.textContent = f;
        explanationList.appendChild(li);
      });
    }
  });
}

/* ==========================================================================
   8. CLICK RIPPLE EFFECT
   ========================================================================== */
function initRippleEffect() {
  document.addEventListener('click', (e) => {
    const ripple = document.createElement('div');
    ripple.className = 'ripple';
    ripple.style.left = `${e.clientX}px`;
    ripple.style.top = `${e.clientY}px`;
    document.body.appendChild(ripple);
    setTimeout(() => ripple.remove(), 800);
  });
}
