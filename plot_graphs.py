minimum = [
    (5, 48.484848), (10, 58.333333), (20, 56.060606), (50, 70.454545),
    (100, 75.000000)
]

average = [
    (5, 60.378788), (10, 68.560606), (20, 69.393939), (50, 76.742424),
    (100, 75.000000)
]


maximum = [
    (5, 71.212121), (10, 82.575758), (20, 82.575758), (50, 83.333333),
    (100, 75.000000)
]

import matplotlib.pyplot as plt


fig, ax = plt.subplots()
legends = []
for i, data in enumerate([minimum, average, maximum]):
    x = [val[0] for val in data]
    y = [val[1] for val in data]

    if i == 0:
        dtype = "Minimum"
    elif i == 1:
        dtype = "Average"
    else:
        dtype = "Maximum"

    ax.plot(x, y)
    legends.append(dtype)


ax.legend(legends)
ax.set(xlabel="Train Set percent (%)", ylabel="Accuracy",
       title="Learning Curve")

ax.grid()
fig.savefig("learning_curves.png")
plt.show()


