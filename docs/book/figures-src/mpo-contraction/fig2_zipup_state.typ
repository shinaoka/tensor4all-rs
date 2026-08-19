// Fig 2: zip-up invariant before processing site n:
// finished C_1..C_{n-1} (canonical), carry T, unprocessed A_n.., B_n..
#import "@preview/cetz:0.4.2": canvas, draw
#import "lib.typ": *
#set page(width: auto, height: auto, margin: 3pt)
#set text(size: 11pt)

#canvas(length: 1cm, {
  import draw: *
  let yA = 1.6 // unprocessed A lane
  let yB = -1.6 // unprocessed B lane
  let xC1 = 0
  let xDots = 1.7
  let xCn1 = 3.4
  let xT = 5.6
  let xAn = 7.6
  let xDots2 = 9.2
  let xLast = 10.6

  // Finished part on the center lane
  tensor((xC1, 0), $C_1$, colC)
  content((xDots, 0.02), $dots.c$)
  tensor((xCn1, 0), $C_(n-1)$, colC, w: 1.2)
  line((xC1 + BW / 2, 0), (xDots - 0.45, 0), stroke: legstroke)
  line((xDots + 0.45, 0), (xCn1 - 0.6, 0), stroke: legstroke)
  vleg((xC1, 0), 1, $sigma_1$)
  vleg((xC1, 0), -1, $omega_1$)
  vleg((xCn1, 0), 1, $sigma_(n-1)$)
  vleg((xCn1, 0), -1, $omega_(n-1)$)

  // Carry tensor T with bond mu to C_{n-1}
  tensor((xT, 0), $T$, colT)
  line((xCn1 + 0.6, 0), (xT - BW / 2, 0), stroke: legstroke)
  content(((xCn1 + 0.6 + xT - BW / 2) / 2, 0.12), $mu$, anchor: "south")

  // Diagonal legs a, b from T to the unprocessed lanes
  line((xT + BW / 2, BH / 4), (xAn - BW / 2, yA), stroke: legstroke)
  content(((xT + xAn) / 2 - 0.15, yA / 2 + 0.28), $a$, anchor: "south")
  line((xT + BW / 2, -BH / 4), (xAn - BW / 2, yB), stroke: legstroke)
  content(((xT + xAn) / 2 - 0.15, yB / 2 - 0.28), $b$, anchor: "north")

  // Unprocessed A and B lanes
  tensor((xAn, yA), $A_n$, colA)
  tensor((xAn, yB), $B_n$, colB)
  vleg((xAn, yA), 1, $sigma_n$)
  vleg((xAn, yB), -1, $omega_n$)
  vbond((xAn, yA), (xAn, yB), $tau_n$)

  line((xAn + BW / 2, yA), (xDots2 - 0.45, yA), stroke: legstroke)
  content((xDots2, yA + 0.02), $dots.c$)
  line((xDots2 + 0.45, yA), (xLast - BW / 2, yA), stroke: legstroke)
  tensor((xLast, yA), $A_L$, colA)
  vleg((xLast, yA), 1, $sigma_L$)

  line((xAn + BW / 2, yB), (xDots2 - 0.45, yB), stroke: legstroke)
  content((xDots2, yB + 0.02), $dots.c$)
  line((xDots2 + 0.45, yB), (xLast - BW / 2, yB), stroke: legstroke)
  tensor((xLast, yB), $B_L$, colB)
  vleg((xLast, yB), -1, $omega_L$)
  vbond((xLast, yA), (xLast, yB), $tau_L$)
})
