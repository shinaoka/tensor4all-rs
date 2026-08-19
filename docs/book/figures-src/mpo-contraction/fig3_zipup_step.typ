// Fig 3: one zip-up step at site n:
// (T, A_n, B_n) --contract--> W2 --SVD cut chi--> C_n + new carry T_n
#import "@preview/cetz:0.4.2": canvas, draw
#import "lib.typ": *
#set page(width: auto, height: auto, margin: 3pt)
#set text(size: 11pt)

#canvas(length: 1cm, {
  import draw: *
  let yA = 1.5
  let yB = -1.5
  let arrowstroke = rgb("#37424a") + 1.2pt

  // ---- Panel 1: carry + A_n + B_n ----
  let xT = 0
  let xAB = 1.9
  tensor((xT, 0), $T$, colT)
  hleg((xT, 0), -1, $mu$)
  line((xT + BW / 2, BH / 4), (xAB - BW / 2, yA), stroke: legstroke)
  content(((xT + xAB) / 2 - 0.15, yA / 2 + 0.25), $a$, anchor: "south")
  line((xT + BW / 2, -BH / 4), (xAB - BW / 2, yB), stroke: legstroke)
  content(((xT + xAB) / 2 - 0.15, yB / 2 - 0.25), $b$, anchor: "north")
  tensor((xAB, yA), $A_n$, colA)
  tensor((xAB, yB), $B_n$, colB)
  vleg((xAB, yA), 1, $sigma$)
  vleg((xAB, yB), -1, $omega$)
  vbond((xAB, yA), (xAB, yB), $tau$)
  hleg((xAB, yA), 1, $a'$)
  hleg((xAB, yB), 1, $b'$)

  // ---- Arrow: contract ----
  line((3.9, 0), (5.0, 0), stroke: arrowstroke, mark: (end: "stealth", fill: rgb("#37424a")))
  content((4.45, 0.25), [contract], anchor: "south")

  // ---- Panel 2: W2 with 5 legs ----
  let xW = 6.8
  tensor((xW, 0), $W_2$, colT, w: 1.15, h: 0.85)
  hleg((xW, 0), -1, $mu$, w: 1.15)
  vleg((xW, 0), 1, $sigma$, h: 0.85)
  vleg((xW, 0), -1, $omega$, h: 0.85)
  line((xW + 1.15 / 2, 0.2), (xW + 1.15 / 2 + LEG, 0.65), stroke: legstroke)
  content((xW + 1.15 / 2 + LEG + 0.12, 0.72), $a'$, anchor: "west")
  line((xW + 1.15 / 2, -0.2), (xW + 1.15 / 2 + LEG, -0.65), stroke: legstroke)
  content((xW + 1.15 / 2 + LEG + 0.12, -0.72), $b'$, anchor: "west")

  // ---- Arrow: SVD ----
  line((9.0, 0), (10.2, 0), stroke: arrowstroke, mark: (end: "stealth", fill: rgb("#37424a")))
  content((9.6, 0.25), [SVD, $lt.eq chi$], anchor: "south")

  // ---- Panel 3: C_n + new carry T_n ----
  let xC = 11.9
  let xTn = 13.9
  tensor((xC, 0), $C_n$, colC)
  hleg((xC, 0), -1, $mu$)
  vleg((xC, 0), 1, $sigma$)
  vleg((xC, 0), -1, $omega$)
  bond((xC, 0), (xTn, 0), $mu'$)
  tensor((xTn, 0), $T_n$, colT)
  line((xTn + BW / 2, 0.2), (xTn + BW / 2 + LEG, 0.65), stroke: legstroke)
  content((xTn + BW / 2 + LEG + 0.12, 0.72), $a'$, anchor: "west")
  line((xTn + BW / 2, -0.2), (xTn + BW / 2 + LEG, -0.65), stroke: legstroke)
  content((xTn + BW / 2 + LEG + 0.12, -0.72), $b'$, anchor: "west")
})
