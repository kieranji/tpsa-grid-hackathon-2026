import sys

sys.path.insert(0, "data/participant-kit")

import gridkit


n = gridkit.load("WP2033", "north-west")
gridkit.solve(n)

a = "5041-17010-2"
b = "1701-5041-1"

print("\n=== LINE DATA ===")
print(
    n.lines.loc[
        [a, b],
        ["bus0", "bus1", "x", "r", "s_nom"]
    ]
)

loading = gridkit.line_loading(n)

t = loading[a].idxmax()

print("\n=== AT TARGET PEAK HOUR ===")
print("snapshot:", t)
print("flow A:", n.lines_t.p0.loc[t, a])
print("flow B:", n.lines_t.p0.loc[t, b])
print("loading A:", loading.loc[t, a])
print("loading B:", loading.loc[t, b])

print("\n=== MAX LOADING ===")
print(loading[[a, b]].max())

print("\n=== FLOW RATIO ===")
print(
    "abs(A/B):",
    abs(
        n.lines_t.p0.loc[t, a]
        / n.lines_t.p0.loc[t, b]
    ),
)
print("\n=== BUS DETAILS ===")

print(
    n.buses.loc[
        ["Srananagh 220", "Cathaleen's Fall"],
        ["v_nom"]
    ]
)