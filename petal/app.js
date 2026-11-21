/* ═══════════════════════════════════════════════════
   PETAL — Flower Search Engine
   app.js  — ML Engine, Search Logic, UI
   ═══════════════════════════════════════════════════ */

/* ──────────────────────────────────────────
   1. FLOATING PETALS ANIMATION
────────────────────────────────────────── */
const PETAL_EMOJIS = ['🌸', '🌺', '🌼', '🌷', '🍃', '✿', '❀', '🌹'];
const petalsContainer = document.getElementById('petals-container');

function spawnPetal() {
  const el = document.createElement('div');
  el.className = 'petal';
  el.textContent = PETAL_EMOJIS[Math.floor(Math.random() * PETAL_EMOJIS.length)];
  el.style.cssText = [
    `left: ${Math.random() * 100}vw`,
    `font-size: ${0.7 + Math.random() * 0.9}rem`,
    `animation-duration: ${7 + Math.random() * 11}s`,
    `animation-delay: ${Math.random() * 2}s`,
  ].join(';');
  petalsContainer.appendChild(el);
  setTimeout(() => el.remove(), 20000);
}

// Spawn initial petals, then drip a new one every 2.8 s
[0, 400, 800].forEach(d => setTimeout(spawnPetal, d));
setInterval(spawnPetal, 2800);


/* ──────────────────────────────────────────
   2. TF-IDF VECTOR ENGINE  (ML core)
────────────────────────────────────────── */

/**
 * Tokenise a string into lower-case alpha-numeric tokens of length ≥ 3.
 * @param {string} text
 * @returns {string[]}
 */
function tokenize(text) {
  return text
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, ' ')
    .split(/\s+/)
    .filter(t => t.length >= 3);
}

/**
 * Build the IDF (inverse document frequency) dictionary from the corpus.
 * Uses add-one (Laplace) smoothing.
 */
function buildIDF(docs) {
  const N = docs.length;
  const df = {};                          // document frequency per term

  docs.forEach(tokens => {
    const unique = new Set(tokens);
    unique.forEach(t => { df[t] = (df[t] || 0) + 1; });
  });

  const idf = {};
  Object.keys(df).forEach(t => {
    idf[t] = Math.log((N + 1) / (df[t] + 1)) + 1;   // smoothed IDF
  });
  return idf;
}

/**
 * Convert a token array into a TF-IDF sparse vector (plain object).
 * @param {string[]} tokens
 * @param {Object}   idf
 * @returns {Object}  { term: weight }
 */
function toTFIDF(tokens, idf) {
  const tf = {};
  tokens.forEach(t => { tf[t] = (tf[t] || 0) + 1; });
  const len = tokens.length || 1;
  const vec = {};
  Object.entries(tf).forEach(([t, count]) => {
    const idfVal = idf[t];
    if (idfVal) vec[t] = (count / len) * idfVal;
  });
  return vec;
}

/**
 * Cosine similarity between two sparse TF-IDF vectors.
 * @returns {number} 0–1
 */
function cosineSim(a, b) {
  const terms = new Set([...Object.keys(a), ...Object.keys(b)]);
  let dot = 0, magA = 0, magB = 0;
  terms.forEach(t => {
    const va = a[t] || 0;
    const vb = b[t] || 0;
    dot  += va * vb;
    magA += va * va;
    magB += vb * vb;
  });
  if (!magA || !magB) return 0;
  return dot / (Math.sqrt(magA) * Math.sqrt(magB));
}

/* ── Build corpus & document vectors at startup ── */
const CORPUS = FLOWER_DB.map(f =>
  tokenize([f.title, f.description, ...f.tags, ...f.keywords].join(' '))
);

const IDF         = buildIDF(CORPUS);
const DOC_VECTORS = CORPUS.map(tokens => toTFIDF(tokens, IDF));

/* Update the DB-size counter once we know the size */
document.getElementById('dbSize').textContent = FLOWER_DB.length;


/* ──────────────────────────────────────────
   3. SEMANTIC RERANKING  (AI layer)
────────────────────────────────────────── */

/**
 * Ask to score each candidate for semantic relevance [0,1].
 * Returns the score array, or null on failure.
 * @param {string}   query
 * @param {Object[]} candidates  — array of FLOWER_DB entries
 * @returns {Promise<number[]|null>}
 */
async function getSemanticBoosts(query, candidates) {
  const prompt =
    `You are a flower encyclopedia semantic-search assistant.\n` +
    `Given a user query and a list of flower names, return ONLY a JSON array of ` +
    `relevance scores (0.0–1.0, same order as candidates, no markdown, no explanation).\n\n` +
    `Query: "${query}"\n` +
    `Candidates: ${JSON.stringify(candidates.map(c => c.title))}`;

  try {
    const res = await fetch('https://api.anthropic.com/v1/messages', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: 'claude-sonnet-4-20250514',
        max_tokens: 300,
        messages: [{ role: 'user', content: prompt }],
      }),
    });
    const data = await res.json();
    const raw  = (data.content || []).map(b => b.text || '').join('').trim();
    const clean = raw.replace(/```json|```/g, '').trim();
    const scores = JSON.parse(clean);
    if (Array.isArray(scores) && scores.length === candidates.length) return scores;
  } catch (_) { /* silently degrade */ }
  return null;
}

/**
 * Ask Claude to identify the most semantically relevant tokens
 * in a flower's description relative to the query.
 * @returns {Promise<Array<{word:string, relevance:'HIGH'|'MID'|'LOW'}>>}
 */
async function getSemanticTokens(query, flower) {
  const snippet = flower.description.slice(0, 140);
  const prompt = //note de pertinence
    `Query: "${query}"\nFlower: "${flower.title} — ${snippet}"\n` +
    `List 5 key words from the flower description most related to the query. ` +
    `Rate each HIGH, MID, or LOW.\n` +
    `Return ONLY a JSON array: [{"word":"example","relevance":"HIGH"}]. No markdown.`;

  try { //On envoie la lettre à Claude par internet
    const res = await fetch('https://api.anthropic.com/v1/messages', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: 'claude-sonnet-4-20250514',
        max_tokens: 180,
        messages: [{ role: 'user', content: prompt }],
      }),
    });
    const data = await res.json();
    const raw  = (data.content || []).map(b => b.text || '').join('').trim();
    return JSON.parse(raw.replace(/```json|```/g, '').trim());
  } catch (_) { return []; }
}


/* ──────────────────────────────────────────
   4. EMBEDDING VISUALISER
────────────────────────────────────────── */

const BAR_PALETTE = [
  '#f2c4c4', '#d98b8b', '#8faa8b', '#c9b8d8',
  '#b07b96', '#e8d5b7', '#b8cdb5', '#c4a882',
];

/**
 * Render a 48-bar chart representing the top 48 TF-IDF weights of a vector.
 * @param {Object} vec  — sparse TF-IDF vector
 */
function renderEmbeddingViz(vec) {
  const vals = Object.values(vec).slice(0, 48);
  const max  = Math.max(...vals, 0.001);

  const barsHTML = vals
    .map((v, i) => {
      const heightPct = Math.max(4, (v / max) * 100);
      const colour    = BAR_PALETTE[i % BAR_PALETTE.length];
      return `<div class="emb-bar"
                   style="height:${heightPct}%;
                          background:${colour};
                          opacity:0.85;
                          animation-delay:${i * 11}ms">
              </div>`;
    })
    .join('');

  document.getElementById('embBars').innerHTML = barsHTML;
  document.getElementById('vectorViz').style.display = 'block';
}


/* ──────────────────────────────────────────
   5. SUGGESTIONS
────────────────────────────────────────── */

const SUGGESTION_CHIPS = [
  'fragrant flowers for weddings 💒',
  'purple calming healing herbs 🌿',
  'bright tropical exotic blooms 🌺',
  'tiny delicate wildflowers 🌾',
  'flowers that climb walls 🧗',
  'easy flowers for beginners 🌱',
  'autumn harvest garden colours 🍂',
  'flowers that attract butterflies 🦋',
  'blue flowers for cottage gardens 💙',
  'rare orchids with amazing scents 🎼',
];

function renderSuggestions() {
  document.getElementById('suggestions').innerHTML =
    SUGGESTION_CHIPS
      .map(s => `<span class="chip" role="button" tabindex="0" aria-label="Search: ${s}"
                       onclick="quickSearch('${s.replace(/'/g, "\\'")}')">
                   ${s}
                 </span>`)
      .join('');
}

/** Populate the search input and fire search */
function quickSearch(query) {
  document.getElementById('searchInput').value = query;
  performSearch();
}


/* ──────────────────────────────────────────
   6. MAIN SEARCH PIPELINE
────────────────────────────────────────── */

/** Show/hide the loading spinner */
function setLoading(visible) {
  document.getElementById('loading').style.display = visible ? 'block' : 'none';
}

async function performSearch() {
  const rawQuery = document.getElementById('searchInput').value.trim();
  if (!rawQuery) return;

  // Reset UI
  document.getElementById('results').innerHTML = '';
  document.getElementById('vectorViz').style.display = 'none';
  setLoading(true);

  /* ── Step 1: TF-IDF vector search ── */
  const queryTokens = tokenize(rawQuery);
  const queryVec    = toTFIDF(queryTokens, IDF);
  renderEmbeddingViz(queryVec);          // show vector visualisation

  const scored = DOC_VECTORS.map((dv, i) => ({
    doc:    FLOWER_DB[i],
    tfidf:  cosineSim(queryVec, dv),
    index:  i,
  }));

  // Take the top-12 candidates for semantic reranking
  const topCandidates = [...scored]
    .sort((a, b) => b.tfidf - a.tfidf)
    .slice(0, 12);

  /* ── Step 2: Claude semantic boost ── */
  const semanticScores = await getSemanticBoosts(rawQuery, topCandidates.map(c => c.doc));

  topCandidates.forEach((c, i) => {
    const boost  = semanticScores ? (semanticScores[i] || 0) : 0.5;
    // Blend: 35% TF-IDF keyword signal + 65% semantic AI
    c.combined       = 0.35 * c.tfidf + 0.65 * boost;
    c.semanticScore  = boost;
  });

  // Re-sort by blended score, filter out very weak matches
  const finalResults = topCandidates
    .sort((a, b) => b.combined - a.combined)
    .filter(r => r.combined > 0.03);

  /* ── Step 3: Semantic token highlights for top 3 ── */
  const tokenPromises = finalResults
    .slice(0, 3)
    .map(r => getSemanticTokens(rawQuery, r.doc));
  const tokenResults = await Promise.all(tokenPromises);

  setLoading(false);
  renderResults(finalResults, tokenResults);
}


/* ──────────────────────────────────────────
   7. RESULTS RENDERER
────────────────────────────────────────── */

/** Choose a tag CSS class by its position in the array */
function tagClass(index) {
  const classes = ['tag-rose', 'tag-sage', 'tag-lav', 'tag-sage'];
  return classes[index % classes.length];
}

/**
 * Build the HTML for a single result card.
 * @param {Object}   result       — { doc, combined, tfidf, semanticScore }
 * @param {number}   animDelay    — ms
 * @param {Array}    tokens       — [{ word, relevance }] or []
 * @returns {string} HTML string
 */
function buildCardHTML(result, animDelay, tokens) {
  const { doc, combined, tfidf, semanticScore } = result;
  const matchPct   = Math.round(combined * 100);
  const tfidfPct   = Math.round(tfidf * 100);
  const semPct     = Math.round((semanticScore || 0) * 100);

  const tagsHTML = doc.tags
    .map((t, i) => `<span class="tag ${tagClass(i)}">${t}</span>`)
    .join('');

  const tokensHTML = tokens.length
    ? `<div class="token-row">
         ${tokens
           .map(t => `<span class="token token-${(t.relevance || 'low').toLowerCase()}">${t.word}</span>`)
           .join('')}
       </div>`
    : '';

  return `
    <article class="result-card" style="animation-delay:${animDelay}ms">
      <div class="card-top">
        <div class="card-left">
          <span class="card-emoji" aria-hidden="true">${doc.emoji}</span>
          <h2 class="card-title">${doc.title}</h2>
          <p  class="card-latin">${doc.latin}</p>
        </div>
        <div class="sim-box" aria-label="Match score: ${matchPct}%">
          <div class="sim-circle">
            <span class="sim-pct">${matchPct}%</span>
            <span class="sim-lbl">match</span>
          </div>
        </div>
      </div>

      <p class="card-desc">${doc.description}</p>

      <div class="card-meta">
        ${tagsHTML}
        <span class="card-type">${doc.type}</span>
      </div>

      <div class="score-details" aria-label="Score breakdown">
        <span>🌿 TF-IDF <span>${tfidfPct}%</span></span>
        <span>🤖 AI Semantic <span>${semPct}%</span></span>
        <span>⭐ Blended <span>${matchPct}%</span></span>
      </div>

      ${tokensHTML}
    </article>`;
}

/**
 * Inject the full results section into the DOM.
 */
function renderResults(results, tokenResults) {
  const el = document.getElementById('results');

  if (!results.length) {
    el.innerHTML = `
      <div class="no-results" role="alert">
        <span class="icon" aria-hidden="true">🌸</span>
        <p>No flowers found for that search — try something like <em>"fragrant pink blooms"</em>!</p>
      </div>`;
    return;
  }

  const header = `
    <div class="results-header">
      <p class="results-count">Found <strong>${results.length}</strong> beautiful flowers</p>
      <p class="sort-note">✨ ranked by AI semantic similarity</p>
    </div>`;

  const cards = results
    .map((r, i) => buildCardHTML(r, i * 75, tokenResults[i] || []))
    .join('');

  el.innerHTML = header + cards;

  // Smooth scroll to first result on mobile
  if (window.innerWidth < 640) {
    el.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }
}


/* ──────────────────────────────────────────
   8. EVENT LISTENERS & INIT
────────────────────────────────────────── */

/* Enter key in the search box */
document.getElementById('searchInput').addEventListener('keydown', e => {
  if (e.key === 'Enter') performSearch();
});

/* Search button */
document.getElementById('searchBtn').addEventListener('click', performSearch);

/* Keyboard accessibility for suggestion chips */
document.getElementById('suggestions').addEventListener('keydown', e => {
  if (e.key === 'Enter' || e.key === ' ') {
    const chip = e.target.closest('.chip');
    if (chip) chip.click();
  }
});

/* Boot: render suggestions and update stat counter */
renderSuggestions();
document.getElementById('dbSize').textContent = FLOWER_DB.length;
