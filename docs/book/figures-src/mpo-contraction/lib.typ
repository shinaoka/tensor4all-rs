// Shared helpers for MPO-MPO contraction tensor diagrams (cetz).
#import "@preview/cetz:0.4.2": canvas, draw

// Named geometry (skill: parameterize, no scattered magic numbers)
#let BW = 1.0 // tensor box width
#let BH = 0.7 // tensor box height
#let LEG = 0.75 // horizontal leg length
#let VLEG = 0.55 // vertical (physical) leg length
#let LANE = 1.05 // vertical distance from center lane to A/B lanes

// Palette (projector-safe, darker mixed colors)
#let colA = (fill: rgb("#dce6f2"), stroke: rgb("#3a5f8f") + 1.1pt)
#let colB = (fill: rgb("#f6e8d8"), stroke: rgb("#a86b2d") + 1.1pt)
#let colC = (fill: rgb("#dfeee5"), stroke: rgb("#2f6f4f") + 1.1pt)
#let colT = (fill: rgb("#eceff1"), stroke: rgb("#5b6770") + 1.1pt)
#let legstroke = rgb("#37424a") + 1.1pt

// A tensor as a rounded box with centered label. pos is (x, y).
#let tensor(pos, label, col, w: BW, h: BH) = {
  import draw: *
  let (x, y) = pos
  rect((x - w / 2, y - h / 2), (x + w / 2, y + h / 2), radius: 0.12, ..col)
  content((x, y), label)
}

// Horizontal leg from box edge, dir: -1 (left) or +1 (right); label above.
#let hleg(pos, dir, label, w: BW, len: LEG) = {
  import draw: *
  let (x, y) = pos
  let x0 = x + dir * w / 2
  let x1 = x0 + dir * len
  line((x0, y), (x1, y), stroke: legstroke)
  content(((x0 + x1) / 2, y + 0.12), label, anchor: "south")
}

// Vertical leg, dir: +1 (up) or -1 (down); label beyond the end.
#let vleg(pos, dir, label, h: BH, len: VLEG) = {
  import draw: *
  let (x, y) = pos
  let y0 = y + dir * h / 2
  let y1 = y0 + dir * len
  line((x, y0), (x, y1), stroke: legstroke)
  content((x, y1 + dir * 0.10), label, anchor: if dir > 0 { "south" } else { "north" })
}

// Bond between two boxes on the same lane, with label above the middle.
#let bond(posl, posr, label, w: BW) = {
  import draw: *
  let (xl, y) = posl
  let (xr, _) = posr
  line((xl + w / 2, y), (xr - w / 2, y), stroke: legstroke)
  content(((xl + xr) / 2, y + 0.12), label, anchor: "south")
}

// Vertical bond between an upper and a lower box (shared physical index).
#let vbond(post, posb, label, h: BH) = {
  import draw: *
  let (x, yt) = post
  let (_, yb) = posb
  line((x, yt - h / 2), (x, yb + h / 2), stroke: legstroke)
  content((x + 0.12, (yt + yb) / 2), label, anchor: "west")
}
