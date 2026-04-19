// =====================================================================
//  Inductive Bias in Biomedical Retrieval -- 5-minute video deck
//
//  Compile locally:   typst compile presentation.typ
//  Or upload to:      https://typst.app
//
//  Dependency (auto-fetched by Typst on first compile):
//    polylux 0.4.0   slide management
// =====================================================================

#import "@preview/polylux:0.4.0": *

// ---- Page + typography ------------------------------------------------
#set page(
  paper: "presentation-16-9",
  margin: (x: 1.6cm, top: 1.2cm, bottom: 1.2cm),
)
#set text(
  font: ("Inter", "Helvetica Neue", "Arial", "New Computer Modern Sans"),
  size: 20pt,
)
#set par(leading: 0.7em)

// ---- Color palette (Innopolis-leaning) --------------------------------
#let accent     = rgb("#B21E32")  // primary red, used for titles + rules
#let pos-color  = rgb("#3C8C5A")  // positive Δ
#let neg-color  = rgb("#C85050")  // negative Δ
#let blue-color = rgb("#3264AA")  // learned models
#let neutral    = rgb("#A8A8A8")  // baseline bars
#let oracle-c   = rgb("#D4A53A")  // oracle bound, distinct from pos
#let bg-soft    = rgb("#F4F2EE")  // subtle fill
#let text-dim   = rgb("#5A5A5A")
#let line-soft  = rgb("#D8D8D8")

// ---- Slide-title helper ----------------------------------------------
#let slide-title(title) = block(below: 0.6em, {
  text(size: 28pt, weight: "bold", fill: accent, title)
  v(-0.35em)
  line(length: 100%, stroke: 1pt + accent)
})

// =====================================================================
// SLIDE 1 -- Title
// =====================================================================
#slide[
  #set page(footer: none)
  #align(horizon + center)[
    #text(size: 38pt, weight: "bold", fill: accent)[
      Inductive Bias in Retrieval
    ]
    #v(0.4em)
    #text(size: 22pt, fill: text-dim)[
      A Multi-Model, Multi-Dataset Study
    ]
    #v(2.5em)
    #text(size: 18pt)[Dmitrii Naumov]
    #v(0.3em)
    #text(size: 16pt, fill: text-dim)[
      Innopolis University · Advanced Machine Learning · Spring 2026
    ]
  ]
]

// =====================================================================
// SLIDE 2 -- The research question
// =====================================================================
#slide[
  #slide-title[Project Idea]

  #v(0.6em)

  #align(center)[
    #text(size: 24pt, weight: "bold")[
      Does the inductive bias of a retriever determine

      its performance on biomedical retrieval?
    ]
    #v(0.6em)
    #text(size: 17pt, style: "italic", fill: text-dim)[
      Or does the right answer depend on the dataset and query type?
    ]
  ]

  #v(1.4em)

  #align(center)[
    #table(
      columns: 2,
      align: (left, left),
      stroke: none,
      inset: (x: 14pt, y: 6pt),
      fill: (col, row) => if row == 0 { bg-soft } else { none },
      [*Bias class*],            [*Example model*],
      [Lexical sparse],          [BM25],
      [General-domain dense],    [MiniLM, BGE-small, E5-small],
      [Learned sparse],          [SPLADE],
      [Domain-adapted dense],    [MedCPT (PubMed-trained)],
    )
  ]

  #align(center)[
    #text(size: 13pt, fill: text-dim)[
      6 retrievers · 5 BEIR-style benchmarks · 3 add-ons
      (cross-encoder reranker, per-query router, downstream RAG)
    ]
  ]
]

// =====================================================================
// SLIDE 3 -- Theoretical framing (card-based layout, no fragile math)
// =====================================================================
#slide[
  #slide-title[Theoretical Framing]

  #v(0.5em)

  #text(size: 17pt)[Three course concepts anchor the analysis.]

  #v(0.9em)

  #let concept-card(num, title, body) = block(
    fill: bg-soft,
    inset: 14pt,
    radius: 6pt,
    stroke: 1pt + accent.lighten(60%),
    width: 100%,
    height: 7.5em,
    breakable: false,
  )[
    #text(size: 30pt, weight: "bold", fill: accent)[#num]
    #h(0.4em)
    #text(size: 18pt, weight: "bold")[#title]

    #v(0.3em)

    #text(size: 13pt)[#body]
  ]

  #grid(
    columns: (1fr, 1fr, 1fr),
    column-gutter: 0.9em,

    concept-card(
      [1],
      [No Free Lunch],
      [No inductive bias is universally optimal. Performance depends on
       alignment between the bias and the data distribution.],
    ),

    concept-card(
      [2],
      [Empirical vs.\ true risk],
      [How well does the test-set estimate of the loss approximate the
       expected risk on the full query distribution?],
    ),

    concept-card(
      [3],
      [Stability],
      [Does the system ranking hold under different query subsets, and
       across different datasets?],
    ),
  )

  #v(1.4em)

  #align(center)[
    #box(
      fill: accent.lighten(88%),
      inset: (x: 18pt, y: 10pt),
      radius: 4pt,
    )[
      #text(size: 14pt)[
        *Inferential tools:*
        paired bootstrap CI · Hoeffding sample-size bound ·
        train--test generalisation gap
      ]
    ]
  ]
]

// =====================================================================
// SLIDE 4 -- What was built
// =====================================================================
#slide[
  #slide-title[What Was Built]

  #v(0.4em)

  #align(center)[
    #table(
      columns: 7,
      align: (left, center, center, center, center, center, center),
      stroke: 0.5pt + line-soft,
      inset: (x: 10pt, y: 6pt),
      fill: (col, row) => if row == 0 { bg-soft } else { none },
      [*Dataset*], [*BM25*], [*MiniLM*], [*BGE*], [*E5*], [*SPLADE*], [*MedCPT*],
      [NFCorpus],      [✓], [✓], [✓], [✓], [✓], [✓],
      [BioASQ-subset], [✓], [✓], [✓], [✓], [✓], [✓],
      [TREC-COVID],    [✓], [✓], [✓], [✓], [—], [—],
      [SciFact],       [✓], [✓], [✓], [✓], [✓], [✓],
      [ArguAna],       [✓], [✓], [✓], [✓], [✓], [✓],
    )
  ]

  #v(1.2em)

  #text(size: 17pt)[*Three additions on top of the first stage:*]


  #set list(spacing: 0.5em, indent: 1em)
  - BGE cross-encoder reranker, plus in-domain MedCPT cross-encoder
  - Per-query router (logistic + LightGBM, six query features)
  - Downstream PubMedQA QA with flan-t5-base (with retrieval ablation)

  #align(center)[
    #text(size: 10pt, fill: text-dim, style: "italic")[
      All experiments fit in a single Google Colab T4 session.
    ]
  ]
]

// =====================================================================
// SLIDE 5 -- Result 1: Multi-dataset heatmap
// =====================================================================
#slide[
  #slide-title[Result 1: No Single Winner]

  #v(0.4em)
  nDCG\@10
  #align(center)[
    #table(
      columns: 7,
      align: (left, center, center, center, center, center, center),
      stroke: 0.5pt + line-soft,
      inset: (x: 9pt, y: 6pt),
      fill: (col, row) => if row == 0 { bg-soft } else { none },
      [*Dataset*], [*BM25*], [*MiniLM*], [*BGE*], [*E5*], [*SPLADE*], [*MedCPT*],

      [NFCorpus],
      [0.294], [0.317],
      table.cell(fill: pos-color.lighten(70%))[*0.339*],
      [0.327], [0.333], [0.325],

      [BioASQ-subset#super[]],
      table.cell(fill: pos-color.lighten(70%))[*0.754*],
      [0.539], [0.684], [0.666], [0.673], [0.610],

      [TREC-COVID],
      [0.576], [0.454], [0.645],
      table.cell(fill: pos-color.lighten(70%))[*0.724*],
      [—], [—],

      [SciFact],
      [0.662], [0.645],
      table.cell(fill: pos-color.lighten(70%))[*0.720*],
      [0.688], [0.633], [0.710],

      [ArguAna],
      [0.361], [0.370],
      table.cell(fill: pos-color.lighten(70%))[*0.429*],
      [0.310], [0.394],
      table.cell(fill: neg-color.lighten(70%))[*0.133*],
    )
  ]

  #v(0.9em)

  #set list(spacing: 0.4em, indent: 0.6em)
  #text(size: 15pt)[
    - BGE-small wins 3 datasets, E5-small wins 1, BM25 wins 1.
    - MedCPT collapses on ArguAna ($-0.227$ vs.\ BM25): biomedical
      pre-training actively hurts on a non-biomedical task.
    - #super[] BioASQ-subset corpus has zero distractors $arrow$ BM25 inflated;
      treat as control, not as biomedical evidence.
  ]
]

// =====================================================================
// SLIDE 6 -- Result 2: Reranker + Router bars (typst-native, no cetz)
// =====================================================================
#slide[
  #slide-title[Result 2: Stronger Reranker $eq.not$ Stronger Pipeline]

  // ---- Bar-row helpers (no external packages) ----------
  // Each chart is a 3-column grid: label | bar area | value.
  // The bar area is a fixed-width box; the bar itself is a child box
  // positioned by the typst `place` function.

  // Reranker chart -- centred zero line, bars grow left or right.
  #let RR-CHART-W = 8.4cm
  #let RR-NEG-MAX = 0.27
  #let RR-POS-MAX = 0.10
  #let RR-ZERO-X  = RR-NEG-MAX / (RR-NEG-MAX + RR-POS-MAX) * RR-CHART-W
  #let RR-BAR-H   = 13pt

  #let rr-row(label, value, vtext, color) = grid(
    columns: (3.4cm, RR-CHART-W, 1.4cm),
    align: (right + horizon, left + horizon, left + horizon),
    inset: (y: 0pt),

    text(size: 8pt, label),

    block(width: RR-CHART-W, height: RR-BAR-H, breakable: false)[
      // Dashed zero line
      #place(top + left, dx: RR-ZERO-X)[
        #line(
          start: (0pt, 0pt), end: (0pt, RR-BAR-H),
          stroke: (paint: rgb("#9A9A9A"), dash: "dashed", thickness: 0.5pt),
        )
      ]
      // The bar itself
      #if value < 0 {
        let w = calc.abs(value) / (RR-NEG-MAX + RR-POS-MAX) * RR-CHART-W
        place(top + left, dx: RR-ZERO-X - w)[
          #block(width: w, height: RR-BAR-H, fill: color)
        ]
      } else if value > 0 {
        let w = value / (RR-NEG-MAX + RR-POS-MAX) * RR-CHART-W
        place(top + left, dx: RR-ZERO-X)[
          #block(width: w, height: RR-BAR-H, fill: color)
        ]
      }
    ],

    text(size: 9pt, fill: text-dim, vtext),
  )

  // Router chart -- bars grow rightward from a value-axis baseline.
  #let RT-CHART-W = 8.4cm
  #let RT-MIN     = 0.28
  #let RT-MAX     = 0.43
  #let RT-RANGE   = RT-MAX - RT-MIN
  #let RT-BAR-H   = 16pt

  #let rt-row(label, value, vtext, color) = grid(
    columns: (3.4cm, RT-CHART-W, 1.4cm),
    align: (right + horizon, left + horizon, left + horizon),
    inset: (y: 0pt),

    text(size: 9pt, label),

    block(width: RT-CHART-W, height: RT-BAR-H, breakable: false)[
      #let w = (value - RT-MIN) / RT-RANGE * RT-CHART-W
      #place(top + left)[
        #block(width: w, height: RT-BAR-H, fill: color)
      ]
    ],

    text(size: 9pt, fill: text-dim, vtext),
  )

  #grid(
    columns: (1fr),
    column-gutter: 0.9em,
    align: top,

    // ---------- LEFT: reranker delta bars ----------
    [
      #align(center)[#text(size: 12pt, weight: "bold")[Reranker $Delta$ nDCG\@10]]

      #rr-row([ArguAna BGE+rer],     -0.243, "-0.243", neg-color)
      #rr-row([BioASQ BM25+rer],     -0.046, "-0.046", neg-color)
      #rr-row([NFC BGE+rer],         -0.039, "-0.039", neg-color)
      #rr-row([SciFact BGE+rer],     -0.011, "-0.011", neg-color)
      #rr-row([NFC BM25+rer],         0.001, "+0.001", neutral)
      #rr-row([NFC MedCPT-CE],        0.027, "+0.027", pos-color)
      #rr-row([TREC-COV E5+rer],      0.030, "+0.030", pos-color)
      #rr-row([BioASQ MedCPT-CE],     0.051, "+0.051", pos-color)

      #align(center)[
        #text(size: 8pt, style: "italic", fill: text-dim)[
          BGE-rerank hurts on 4 of 5 datasets.\
          Domain-aligned MedCPT-CE wins on biomedical sets.
        ]
      ]
    ],

    // ---------- RIGHT: router results ----------
    [
      #align(center)[#text(size: 16pt, weight: "bold")[Router results (NFCorpus)]]

      #rt-row([Always BGE+rer],     0.301, "0.301", neutral)
      #rt-row([Always BM25],        0.308, "0.308", neutral)
      #rt-row([Logistic router],    0.323, "0.323", blue-color)
      #rt-row([LightGBM router],    0.333, "0.333", blue-color)
      #rt-row([Always BGE],         0.339, "0.339", neutral)
      #rt-row([Static Hybrid],      0.343, "0.343", neutral)
      #rt-row([Oracle (UB)],        0.410, "0.410", oracle-c)

    ],
  )

  // Small read-only note: where Δ(q) and the router's six features come from.
  #align(center)[
    #block(
      fill: bg-soft,
      inset: (x: 12pt, y: 7pt),
      radius: 4pt,
      width: 96%,
    )[
      #text(size: 10pt)[
        *Δ(q) = nDCG\@10*#sub[Dense]*(q) − nDCG\@10*#sub[BM25]*(q)*, the
        per-query advantage of Dense over BM25 on NFCorpus. An OLS
        regression on three standardised query features finds two
        significant predictors —
        *vocabulary gap* (β = +0.048, p < 0.001) raises Δ;
        *technicality / mean IDF* (β = −0.034, p = 0.019) lowers it;
        query length is not significant. R² = 0.071.
        The router uses these three features plus three more
        (BM25 top-1 score, question-form flag, medical-token share)
        to predict the best strategy per query.
      ]
    ]
  ]
]

// =====================================================================
// SLIDE 7 -- Take-aways and limitations
// =====================================================================
#slide[
  #slide-title[Take-Aways and Limitations]

  #v(0.3em)

  #text(size: 18pt)[*Three findings:*]

  #v(0.4em)

  #set enum(numbering: n => text(weight: "bold", fill: accent)[#n.])
  #set list(spacing: 0.5em)

  + *Inductive bias is dataset-specific.*
    No retriever and no reranker dominate the five datasets.

  + *Domain alignment matters at the cross-encoder stage*, not at the
    bi-encoder stage. MedCPT-CE gives the only clean biomedical gains
    in the study.

  + *Per-query routing is now a method.* The signal is real
    (oracle $+0.066$ above the best static fusion) but six features
    capture only $tilde 7%$ of $Delta(q)$ variance.


  #text(size: 15pt, fill: accent)[*Limitations:*] #h(0.3em)
  #text(size: 15pt)[
    BioASQ-subset corpus has zero distractors (degenerate);
  ]

]
