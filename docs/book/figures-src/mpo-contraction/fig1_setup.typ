// Fig 1: MPO-MPO product at one site: A_n over B_n (contract tau) = C_n
#import "@preview/cetz:0.4.2": canvas, draw
#import "lib.typ": *
#set page(width: auto, height: auto, margin: 3pt)
#set text(size: 11pt)

#canvas(length: 1cm, {
  import draw: *
  let yA = LANE
  let yB = -LANE
  let xL = 0

  // Left side: A_n over B_n
  tensor((xL, yA), $A_n$, colA)
  tensor((xL, yB), $B_n$, colB)
  hleg((xL, yA), -1, $a$)
  hleg((xL, yA), 1, $a'$)
  hleg((xL, yB), -1, $b$)
  hleg((xL, yB), 1, $b'$)
  vleg((xL, yA), 1, $sigma_n$)
  vleg((xL, yB), -1, $omega_n$)
  vbond((xL, yA), (xL, yB), $tau_n$)

  // equals sign
  content((2.3, 0), $=$)

  // Right side: fused C_n
  let xR = 4.6
  tensor((xR, 0), $C_n$, colC)
  hleg((xR, 0), -1, $(a b)$)
  hleg((xR, 0), 1, $(a' b')$)
  vleg((xR, 0), 1, $sigma_n$)
  vleg((xR, 0), -1, $omega_n$)
})
