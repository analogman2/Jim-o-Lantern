#!/usr/bin/env python3
"""Jim's all-in-one outdoor box: WLED controller, Raspberry Pi 4B + amp HAT, USB mic array,
speaker, and optional Pi camera, sitting on top of a 1 x 1 inch post.

Parametric: change the numbers below, then run
    ../.venv-cad/bin/python jim_case.py
to regenerate the STL/STEP files in out/. Values marked MEASURE are placeholders; measure
your parts and update them before printing.

Print in ASA or PETG (not PLA: it warps in sun and heat). The body prints upright on its
floor; the lid prints upside down (top on the bed). No supports needed except the camera hood.
Heat-set inserts: M3 (corner posts, speaker, camera, socket), M2.5 (Pi), M4 (post thumbscrew).
"""
from build123d import (Align, Axis, Box, Cylinder, Ellipse, Polygon, Pos, Rectangle, Rot, Text,
                       export_step, export_stl, extrude, fillet)
import math
import os

MIN = (Align.MIN, Align.MIN, Align.MIN)
CEN = (Align.CENTER, Align.CENTER, Align.MIN)

# ---------------------------------------------------------------- box
W, D, H = 214.0, 130.0, 110.0   # outside of the body (width, depth front-to-back, height)
WALL, FLOOR = 2.6, 3.0
CORNER_R = 6.0

# heat-set insert hole diameters (typical for brass inserts; check your inserts)
INSERT_M3, INSERT_M25, INSERT_M4 = 4.0, 3.6, 5.6
CLEAR_M3 = 3.4

# ---------------------------------------------------------------- parts (MEASURE = check yours)
PI_W, PI_D = 85.0, 56.0                         # Raspberry Pi 4B board (same holes as 3B)
PI_HOLES = [(3.5, 3.5), (61.5, 3.5), (3.5, 52.5), (61.5, 52.5)]
PI_STANDOFF = 6.0

CTRL_W, CTRL_D = 95.0, 60.0                     # WLED controller board         MEASURE
CTRL_HOLES = [(3.5, 3.5), (91.5, 3.5), (3.5, 56.5), (91.5, 56.5)]   # MEASURE
CTRL_STANDOFF = 6.0
CTRL_POS = (WALL + 11, D - WALL - CTRL_D - 4)   # back-left, clear of the corner post
PI_POS = (CTRL_POS[0] + CTRL_W + 2, D - WALL - PI_D - 4)   # back-right, beside the controller

SPK_X, SPK_Z = 46.0, 42.0                       # speaker centre on the front wall
SPK_CUTOUT = 46.0                               # grille diameter (driver cone)   MEASURE
SPK_FRAME = 53.0                                # driver frame size (square)       MEASURE
SPK_SCREW_SPACING = 58.0                        # cup screws, square pattern
SPK_DEPTH = 42.0                                # sealed chamber depth behind the driver

CAM_X, CAM_Z = 150.0, 86.0                      # camera lens centre on the front wall
CAM_WINDOW = 12.0
CAM_SCREW_SPACING = 36.0                        # mount screws, either side of the window
CAM_HOLES = [(-10.5, -6.25), (10.5, -6.25), (-10.5, 6.25), (10.5, 6.25)]  # Camera Module 3
CAM_LENS_OFFSET = (0.0, 2.25)                   # lens centre relative to the hole pattern

MIC_D = 70.0                                    # mic array board (round)           MEASURE
MIC_HOLES_R = 26.0                              # its mounting holes, on this radius MEASURE
MIC_PORT_R = 30.0                               # ring of sound holes over the mics
MIC_STANDOFF = 10.0

POST = 25.4 + 0.4                               # 1 x 1 inch post, plus clearance
SOCKET_DEPTH, SOCKET_WALL = 45.0, 4.0
GLAND_HOLE = 12.6                               # PG7 cable glands (5 V in, LED outputs)
GLANDS = [(150.0, 20.0), (170.0, 20.0), (190.0, 20.0)]   # floor, front-right (x, y from front-left)
SOCKET_Y = 40.0                                 # post socket centre (from the front)
SOCKET_SPACING = 36.0                           # its screw pattern (square)

LID_T, LID_OVERHANG, SKIRT, SKIRT_GAP = 4.0, 5.0, 8.0, 0.6

# ---------------------------------------------------------------- the Jim-o'-Lantern graphic
FONT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts", "Creepster-Regular.ttf")
EMBOSS = 1.2                                    # how far the graphic stands out from the wall
LOGO_ICON = (97.0, 40.0)                        # pumpkin icon centre on the front (x, z)
LOGO_TEXT_X, LOGO_TEXT_SIZE = 118.0, 22.0       # "Jim-o'-" / "Lantern", to the right of the icon
LOGO_LINES_Z = (45.0, 21.0)                     # baselines of the two lines
SIDE_TEXT, SIDE_TEXT_SIZE, SIDE_TEXT_Z = "Don't call me Jack!", 14.0, 52.0


def at(x, y, z, shape):
    return Pos(x, y, z) * shape


def pumpkin_icon():
    """A little carved jack-o'-lantern (2D, centred on the origin, about 37 x 36 mm)."""
    p = Ellipse(10, 15) + Pos(-8.5, -0.5) * Ellipse(10, 13.5) + Pos(8.5, -0.5) * Ellipse(10, 13.5)
    p += Pos(0.8, 16.5) * Rot(Z=-12) * Rectangle(4, 7)                     # stem
    p -= Polygon((-10, 1.5), (-3, 1.5), (-6.5, 8), align=None)            # eyes
    p -= Polygon((3, 1.5), (10, 1.5), (6.5, 8), align=None)
    p -= Polygon((-1.6, -2.5), (1.6, -2.5), (0, 0.5), align=None)         # nose
    p -= Polygon((-12, -4.5), (-8.5, -6.5), (-6, -4.6), (-3.5, -6.8), (0, -5.5), (3.5, -6.8),
                 (6, -4.6), (8.5, -6.5), (12, -4.5), (9.5, -9.5), (6.5, -8.5), (4, -11.5),
                 (-4, -11.5), (-6.5, -8.5), (-9.5, -9.5), align=None)       # jagged grin
    return p


def on_front(sketch, x, z):
    """Emboss a 2D sketch onto the front wall (reads left to right, standing out toward -Y)."""
    return Pos(x, 0.3, z) * Rot(X=90) * extrude(sketch, EMBOSS + 0.3)


def on_right_side(sketch, y, z):
    """Emboss a 2D sketch onto the right-hand wall (reads front to back, standing out toward +X)."""
    return Pos(W - 0.3, y, z) * Rot(Z=90) * Rot(X=90) * extrude(sketch, EMBOSS + 0.3)


def logo():
    """The raised graphic, in body coordinates. Exported on its own too, so an AMS printer can
    print it in a second colour (load body_plain + logo as one object, two filaments)."""
    g = on_front(pumpkin_icon(), *LOGO_ICON)
    for line, z in zip(("Jim-o'-", "Lantern"), LOGO_LINES_Z):
        g += on_front(Text(line, LOGO_TEXT_SIZE, font_path=FONT, align=(Align.MIN, Align.MIN)),
                      LOGO_TEXT_X, z)
    side = Text(SIDE_TEXT, SIDE_TEXT_SIZE, font_path=FONT, align=(Align.CENTER, Align.CENTER))
    g += on_right_side(side, D / 2, SIDE_TEXT_Z)
    return g


def body():
    return body_plain() + logo()


def body_plain():
    outer = fillet(Box(W, D, H, align=MIN).edges().filter_by(Axis.Z), CORNER_R)
    inner = at(WALL, WALL, FLOOR, Box(W - 2 * WALL, D - 2 * WALL, H, align=MIN))
    b = outer - inner

    # corner posts with M3 inserts for the lid
    for x, y in [(WALL + 5, WALL + 5), (W - WALL - 5, WALL + 5),
                 (WALL + 5, D - WALL - 5), (W - WALL - 5, D - WALL - 5)]:
        b += at(x, y, FLOOR, Cylinder(5, H - FLOOR, align=CEN))
        b -= at(x, y, H - 8, Cylinder(INSERT_M3 / 2, 8.1, align=CEN))

    # Pi and controller standoffs
    for (ox, oy), holes, h, ins in [(PI_POS, PI_HOLES, PI_STANDOFF, INSERT_M25),
                                    (CTRL_POS, CTRL_HOLES, CTRL_STANDOFF, INSERT_M25)]:
        for hx, hy in holes:
            b += at(ox + hx, oy + hy, FLOOR, Cylinder(3.5, h, align=CEN))
            b -= at(ox + hx, oy + hy, FLOOR + h - 5, Cylinder(ins / 2, 5.1, align=CEN))

    # speaker grille: hex pattern of 3 mm holes through the front wall
    pitch, r_hole = 5.0, 1.5
    rows = int(SPK_CUTOUT / (pitch * 0.866)) + 2
    for row in range(-rows, rows + 1):
        for col in range(-rows, rows + 1):
            x = col * pitch + (pitch / 2 if row % 2 else 0)
            z = row * pitch * 0.866
            if math.hypot(x, z) <= SPK_CUTOUT / 2 - r_hole:
                b -= at(SPK_X + x, -1, SPK_Z + z, Rot(X=-90) * Cylinder(r_hole, WALL + 2, align=CEN))
    # bosses for the speaker cup screws, inside the front wall
    s = SPK_SCREW_SPACING / 2
    for dx, dz in [(-s, -s), (s, -s), (-s, s), (s, s)]:
        b += at(SPK_X + dx, WALL, SPK_Z + dz, Rot(X=-90) * Cylinder(4, 8, align=CEN))
        b -= at(SPK_X + dx, WALL + 2, SPK_Z + dz, Rot(X=-90) * Cylinder(INSERT_M3 / 2, 6.1, align=CEN))

    # camera window, hood (keeps rain and glare off the lens), and mount bosses
    b -= at(CAM_X, -1, CAM_Z, Rot(X=-90) * Cylinder(CAM_WINDOW / 2, WALL + 2, align=CEN))
    hood = at(CAM_X, -14, CAM_Z, Rot(X=-90) * Cylinder(14, 14, align=CEN))
    hood -= at(CAM_X, -15, CAM_Z, Rot(X=-90) * Cylinder(12, 16, align=CEN))
    hood -= at(CAM_X - 20, -20, CAM_Z - 30, Box(40, 30, 30, align=MIN))   # keep the top half
    b += hood
    for dx in (-CAM_SCREW_SPACING / 2, CAM_SCREW_SPACING / 2):
        b += at(CAM_X + dx, WALL, CAM_Z, Rot(X=-90) * Cylinder(4, 7, align=CEN))
        b -= at(CAM_X + dx, WALL + 1, CAM_Z, Rot(X=-90) * Cylinder(INSERT_M3 / 2, 6.1, align=CEN))

    # floor: cable glands, vents, and insert bosses for the post socket
    for gx, gy in GLANDS:
        b -= at(gx, gy, -1, Cylinder(GLAND_HOLE / 2, FLOOR + 2, align=CEN))
    for i in range(6):   # vent slots under the Pi: cool air in...
        b -= at(PI_POS[0] + 12 + i * 12, PI_POS[1] + 8, -1,
                Box(4, PI_D - 16, FLOOR + 2, align=(Align.CENTER, Align.MIN, Align.MIN)))
    # ...and warm air out high on the back wall behind the Pi (a Pi 4 runs warm). Narrow
    # horizontal slots, just under the lid's drip skirt so rain mostly can't reach them.
    for i in range(4):
        b -= at(PI_POS[0] + 6, D - WALL - 1, H - SKIRT - 6 - i * 6,
                Box(PI_W - 12, WALL + 2, 2.5, align=(Align.MIN, Align.MIN, Align.CENTER)))
    h = SOCKET_SPACING / 2
    for dx, dy in [(-h, -h), (h, -h), (-h, h), (h, h)]:
        x, y = W / 2 + dx, SOCKET_Y + dy
        b += at(x, y, FLOOR, Cylinder(4.5, 6, align=CEN))
        b -= at(x, y, -1, Cylinder(INSERT_M3 / 2, 8, align=CEN))
    return b


def lid():
    """Printed upside down: the top surface is on the bed (z=0), skirt and standoffs face up."""
    lw, ld = W + 2 * LID_OVERHANG, D + 2 * LID_OVERHANG
    plate = fillet(Box(lw, ld, LID_T, align=MIN).edges().filter_by(Axis.Z), CORNER_R + LID_OVERHANG)
    # drip skirt around the outside of the body walls
    skirt_out = fillet(Box(lw, ld, SKIRT, align=MIN).edges().filter_by(Axis.Z),
                       CORNER_R + LID_OVERHANG)
    skirt_in = at(LID_OVERHANG - SKIRT_GAP, LID_OVERHANG - SKIRT_GAP, -1,
                  Box(W + 2 * SKIRT_GAP, D + 2 * SKIRT_GAP, SKIRT + 2, align=MIN))
    l = plate + at(0, 0, LID_T, skirt_out - skirt_in)
    ox, oy = LID_OVERHANG, LID_OVERHANG   # body origin in lid coordinates
    # screw holes into the corner posts (countersunk from the top)
    for x, y in [(WALL + 5, WALL + 5), (W - WALL - 5, WALL + 5),
                 (WALL + 5, D - WALL - 5), (W - WALL - 5, D - WALL - 5)]:
        l -= at(ox + x, oy + y, -1, Cylinder(CLEAR_M3 / 2, LID_T + 2, align=CEN))
        l -= at(ox + x, oy + y, -0.01, Cylinder(3.2, 1.8, align=CEN))
    # mic: ring of sound holes plus standoffs; mic sits under the lid near the back
    mx, my = ox + W / 2, oy + D * 0.62
    for i in range(24):
        a = 2 * math.pi * i / 24
        l -= at(mx + MIC_PORT_R * math.cos(a), my + MIC_PORT_R * math.sin(a), -1,
                Cylinder(1.0, LID_T + 2, align=CEN))
    l -= at(mx, my, -1, Cylinder(1.0, LID_T + 2, align=CEN))
    for i in range(4):
        a = math.pi / 4 + i * math.pi / 2
        x, y = mx + MIC_HOLES_R * math.cos(a), my + MIC_HOLES_R * math.sin(a)
        l += at(x, y, LID_T, Cylinder(3.5, MIC_STANDOFF, align=CEN))
        l -= at(x, y, LID_T + MIC_STANDOFF - 5, Cylinder(INSERT_M25 / 2, 5.1, align=CEN))
    return l


def speaker_cup():
    """Sealed back chamber: clamps the driver against the front wall (add a foam gasket)."""
    size = SPK_SCREW_SPACING + 12
    flange = fillet(Box(size, size, 3, align=CEN).edges().filter_by(Axis.Z), 5)
    cup_out = Box(SPK_FRAME + 8, SPK_FRAME + 8, SPK_DEPTH, align=CEN)
    cup_in = at(0, 0, -1, Box(SPK_FRAME + 1, SPK_FRAME + 1, SPK_DEPTH - 1.5, align=CEN))
    c = flange + cup_out - cup_in
    c -= at(0, 0, -1, Box(SPK_FRAME + 1, SPK_FRAME + 1, 5, align=CEN))
    s = SPK_SCREW_SPACING / 2
    for dx, dy in [(-s, -s), (s, -s), (-s, s), (s, s)]:
        c -= at(dx, dy, -1, Cylinder(CLEAR_M3 / 2, 5, align=CEN))
    c -= at(0, SPK_FRAME / 2 + 1, SPK_DEPTH - 8, Rot(X=-90) * Cylinder(1.6, 6, align=CEN))  # wire hole
    return c


def camera_mount():
    """Holds a Camera Module 3 behind the window; screws onto the two front-wall bosses."""
    span = CAM_SCREW_SPACING + 10
    m = fillet(Box(span, 30, 3, align=CEN).edges().filter_by(Axis.Z), 4)
    for dx in (-CAM_SCREW_SPACING / 2, CAM_SCREW_SPACING / 2):
        m -= at(dx, 0, -1, Cylinder(CLEAR_M3 / 2, 5, align=CEN))
    lx, ly = CAM_LENS_OFFSET
    m -= at(0, 0, -1, Cylinder(5, 5, align=CEN))          # lens opening (lens sits here)
    for hx, hy in CAM_HOLES:                               # camera board standoffs
        m += at(hx - lx, hy - ly, 3, Cylinder(2.2, 3, align=CEN))
        m -= at(hx - lx, hy - ly, 0.5, Cylinder(0.9, 6, align=CEN))   # M2 self-tapping
    return m


def post_socket():
    """Slides onto the 1 x 1 post; screws to the floor bosses; M4 thumbscrew locks it."""
    outer = POST + 2 * SOCKET_WALL
    fl = SOCKET_SPACING + 14
    flange = fillet(Box(fl, fl, 4, align=CEN).edges().filter_by(Axis.Z), 5)
    tube = Box(outer, outer, SOCKET_DEPTH, align=CEN) - at(0, 0, -1, Box(POST, POST, SOCKET_DEPTH - 3, align=CEN))
    s = (flange + at(0, 0, 0, tube))
    h = SOCKET_SPACING / 2
    for dx, dy in [(-h, -h), (h, -h), (-h, h), (h, h)]:
        s -= at(dx, dy, -1, Cylinder(CLEAR_M3 / 2, 6, align=CEN))
    s -= at(outer / 2 - SOCKET_WALL - 1, 0, SOCKET_DEPTH / 2,
            Rot(Y=90) * Cylinder(INSERT_M4 / 2, SOCKET_WALL + 2, align=CEN))
    return s


PARTS = {"body": body, "lid": lid, "speaker_cup": speaker_cup,
         "camera_mount": camera_mount, "post_socket": post_socket,
         # two-colour printing: body_plain + logo, loaded together as one object
         "body_plain": body_plain, "logo": logo}

if __name__ == "__main__":
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
    os.makedirs(out, exist_ok=True)
    for name, make in PARTS.items():
        part = make()
        export_stl(part, os.path.join(out, f"{name}.stl"))
        export_step(part, os.path.join(out, f"{name}.step"))
        bb = part.bounding_box()
        print(f"{name:13} {bb.size.X:6.1f} x {bb.size.Y:6.1f} x {bb.size.Z:6.1f} mm")
