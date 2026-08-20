// Fig 4: 2-site fit local update: Theta = E^L . A_i B_i . A_j B_j . E^R
#import "@preview/cetz:0.4.2": canvas, draw
#import "lib.typ": *
#set page(width: auto, height: auto, margin: 3pt)
#set text(size: 11pt)

#canvas(length: 1cm, {
  import draw: *
  let yA = 1.5
  let yB = -1.5
  let envW = 1.1
  let envH = 4.2
  let xL = 0
  let xi = 2.4
  let xj = 4.9
  let xR = 7.3

  // Left environment (contains conj(C) of the left subtree)
  tensor((xL, 0), $E^L_(i-1)$, colT, w: envW, h: envH)
  vleg((xL, 0), -1, $mu$, h: envH)

  // Site i
  tensor((xi, yA), $A_i$, colA)
  tensor((xi, yB), $B_i$, colB)
  line((xL + envW / 2, yA), (xi - BW / 2, yA), stroke: legstroke)
  content(((xL + envW / 2 + xi - BW / 2) / 2, yA + 0.12), $a$, anchor: "south")
  line((xL + envW / 2, yB), (xi - BW / 2, yB), stroke: legstroke)
  content(((xL + envW / 2 + xi - BW / 2) / 2, yB + 0.12), $b$, anchor: "south")
  vleg((xi, yA), 1, $sigma_i$)
  vleg((xi, yB), -1, $omega_i$)
  vbond((xi, yA), (xi, yB), $tau_i$)

  // Site j = i+1
  tensor((xj, yA), $A_j$, colA)
  tensor((xj, yB), $B_j$, colB)
  bond((xi, yA), (xj, yA), $a'$)
  bond((xi, yB), (xj, yB), $b'$)
  vleg((xj, yA), 1, $sigma_j$)
  vleg((xj, yB), -1, $omega_j$)
  vbond((xj, yA), (xj, yB), $tau_j$)

  // Right environment
  line((xj + BW / 2, yA), (xR - envW / 2, yA), stroke: legstroke)
  content(((xj + BW / 2 + xR - envW / 2) / 2, yA + 0.12), $a''$, anchor: "south")
  line((xj + BW / 2, yB), (xR - envW / 2, yB), stroke: legstroke)
  content(((xj + BW / 2 + xR - envW / 2) / 2, yB + 0.12), $b''$, anchor: "south")
  tensor((xR, 0), $E^R_(j+1)$, colT, w: envW, h: envH)
  vleg((xR, 0), -1, $nu$, h: envH)
})
