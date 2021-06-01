import argparse
from collections.abc import Iterable

import pprint
import random
import sys
import time
from typing import List, Tuple

import matplotlib.pyplot as plt
from matplotlib.text import Annotation
from matplotlib.transforms import IdentityTransform
from matplotlib.patches import Rectangle, Arc
from matplotlib.collections import PatchCollection
import numpy as np


def yield_fib_numbers(n: int) -> Iterable:
    """Short summary.

    Parameters
    ----------
    n : int
        Number of squares to generate.

    Returns
    -------
    Iterable[int]
        Yields a sequence of fibonacci numbers.

    """
    a, b = 1, 1
    for _ in range(n):
        yield a
        a, b = b, (a + b)


def plot_fibonacci_spiral(
    number_of_squares: int,
    draw_square_number_labels: bool = True,
    draw_fibonacci_arcs: bool = True,
    line_width: int = 8,
    color_map: str = "Blues",
    alpha: float = 1,
    draw_golden_ratio: bool = False,
    square_recursion_depth: int = 0,
    fib_list_order: str = "NORMAL",
    output_image_file_path: str = "plot.png"
):
    """Short summary.

    Parameters
    ----------
    number_of_squares : int
        number of squares to draw.
    draw_square_number_labels :
        should square numbers be drawn?
    draw_fibonacci_arcs : bool
        should fibonacci arcs be drawn?.
    line_width: int
        rectangle width.
    color_map : str
        name of the matplotlib color map to use.
        https://matplotlib.org/stable/tutorials/colors/colormaps.html#sphx-glr-tutorials-colors-colormaps-py
    alpha : float
        color transparency to use.
    draw_golden_ratio : bool
        should the golden ratio approximation be displayed?.
    fib_list_order : str
        "NORMAL" is the default fibonacci number sequence, "REVERSE" is as the name implies and "RANDOMIZED"
        shuffles the order.
    output_image_file_path : str
        output file path for the resulting image.

    Returns
    -------
    type
        Description of returned object.

    """
    fibs: List[int] = list(yield_fib_numbers(number_of_squares))

    if fib_list_order == "REVERSE":
        fibs = list(reversed(fibs))
    elif fib_list_order == "RANDOMIZE":
        random.shuffle(fibs)

    pprint.pprint(fibs)



    x: int = 0
    y: int = 0
    angle: int = 180
    center: Tuple[int, int] = (x + fibs[0], y + fibs[0])
    rectangles: List[Rectangle] = []
    xs: List[int] = []
    ys: List[int] = []

    # fig, ax = plt.subplots(1, figsize=(16, 16))
    fig_sz: int = 45 # 32
    fig, ax = plt.subplots(1, figsize=(fig_sz, fig_sz))
    plt.margins(0)

    ##### COMMENT OUT THESE WHEN FINISHED DEVELOPING
    # test_rect: Rectangle = Rectangle((0, 0), 100, 100)
    # rectangles.append(test_rect)
    # test_rect.set_x(test_rect.get_x() + 100)
    #####

    # Switch the axes coordinate limits to match the figure size in pixels.
    ax.set_xlim(0, fig_sz * 100)
    ax.set_ylim(0, fig_sz * 100)

    # Invert Y axis so that our coordinate system matches other projects.
    # See: https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.invert_yaxis.html
    ax.invert_yaxis()
    #rectangles.append(Rectangle((0, 0), 500, 500))

    # Scale the fibonacci sequence numbers to match the axis scaling.
    for i in range(0, len(fibs)):
        # fibs[i] = fibs[i] / len(fibs)
        fibs[i] = fibs[i] * fig_sz * 100 / fibs[-1]
        # fibs[i] /= 2
    # pprint.pprint("> fibs post-scaling:\n{}".format(fibs))


    # TESTING
    # rectangles.append(Rectangle((2700, 2700), 500, 500))

    #####

    drawn_number_font_sz: int = fig_sz / 2 # int(fig_sz * 0.75)

    previous_side: int = fibs[0]
    last_center: Tuple[float, float]

    def add_fib_rectangles(fibs: List[int], x: int, y: int, angle: int, center: Tuple[int, int]) -> None:
        for i, side in enumerate(fibs):

            # side = int(side / 4)

            pprint.pprint("> {}-th Rectangle center: {} width: {} height: {}".format(i, [x, y], side, side))

            # On the coordinate system of matplotlib.patches.Rectangle:
            # https://matplotlib.org/stable/api/_as_gen/matplotlib.patches.Rectangle.html
            rect_center: Tuple[float, float] = (x, y)
            last_center = rect_center
            rectangles.append(Rectangle(rect_center, side, side))
            # rectangles.append(Rectangle(rect_center, side, side, transform=ax.transAxes))

            # rectangles.append(Rectangle([x, y], side, side, linewidth=line_width))
            if draw_square_number_labels and i > number_of_squares - 8:
                ax.annotate(side, xy=(x + 0.45 * side, y + 0.45 * side), fontsize=drawn_number_font_sz)

            if draw_fibonacci_arcs:
                this_arc = Arc(
                    center,
                    2 * side,
                    2 * side,
                    angle=angle,
                    theta1=0,
                    theta2=90,
                    edgecolor="black",
                    antialiased=True,
                )
                ax.add_patch(this_arc)
            angle += 90

            xs.append(x)
            ys.append(y)
            if i == 0:
                x += side
                center = (x, y + side)
            elif i == 1:
                x -= side
                y += side
                center = (x, y)
            elif i in range(2, i + 1, 4):
                x -= side + previous_side
                y -= previous_side
                center = (x + side + previous_side, y)
            elif i in range(3, i + 1, 4):
                y -= side + previous_side
                center = (x + side + previous_side, y + side + previous_side)
            elif i in range(4, i + 1, 4):
                x += side
                center = (x, y + side + previous_side)
            elif i in range(5, i + 1, 4):
                x -= previous_side
                y += side
                center = (x, y)

            pprint.pprint("> center:\t {}".format(center))
            previous_side = side

    # add_fib_rectangles(fibs, x, y, angle, center)



    # col = PatchCollection(rectangles, alpha=alpha, edgecolor="black")

    # Add randomized rectangles.
    # fig_width, fig_height = plt.gcf().get_size_inches() * plt.gcf().dpi
    #fig_width *= fig.dpi

    print('fig.dpi = {}'.format(fig.dpi))
    print('fig.get_size_inches() = ' + str(fig.get_size_inches()))

    ax.grid(True, which='both')

    # fig_width, fig_height = plt.gcf().get_size()
    # print("> Figure dpi: {} width: {} height: {}".format(plt.gcf().dpi, fig_width, fig_height))

    # Coordinate systems of matplotlib: https://matplotlib.org/2.0.2/users/transforms_tutorial.html
    def add_rectangle(current_rectangles: List[Rectangle], last_rect_center: Tuple[float, float],
                      previous_width: int, previous_height: int, area_details: List[Tuple[int, int, int]],
                      depth: int = 1, prev_dir: str ="start") -> None:

        if depth == 0:
            return

        x: float = last_rect_center[0]
        y: float = last_rect_center[1]

        if prev_dir == "vertical" or (prev_dir == "start" and random.random() < 0.5):
            # Horizontal slash.
            prev_dir = "horizontal"
            next_height: float = previous_height / 2

            new_rect: Rectangle = Rectangle(last_rect_center, previous_width, next_height)
            b = new_rect.get_bbox()  # bbox instance


            # pprint.pprint("> bbox coord:\t{}".format(b.transformed(ax.transData)))
            #pprint.pprint("> bbox coord:\t{}".format(ax.transLimits.transform(b)))

            # pprint.pprint("> bbox coord:\t{}".format(IdentityTransform.transform(b)))

            # data_rect: Rectangle = new_rect.get_patch_transform()
            # pprint.pprint("> new_rect [patch coords]-->[data coords]:\t{}".format(data_rect))
            # display_rect = ax.transData.transform(data_rect.xy)
            # pprint.pprint("> new_rect [data coords]-->[display coords]:\t{}".format(display_rect))



            current_rectangles.append(new_rect)
            # if draw_square_number_labels:
            #    ax.annotate(depth, xy=(x + 0.49 * previous_width, y + 0.49 * next_height), fontsize=drawn_number_font_sz)

            if depth > 0:
                if random.random() < 0.5:
                    # We are moving above the horizontal slash.
                    next_center: Tuple[float, float] = (x, y + next_height)

                    xdisplay, ydisplay = ax.transData.transform((x + 0.49 * previous_width, y + 0.49 * next_height))
                    #xdisplay = * fig.dpi
                    #coord_display: str = "({}, {})".format(int(xdisplay), int(fig_sz * 100 - ydisplay))
                    coord_display: str = "{}".format(depth)
                    area_details.append((depth, int(xdisplay), int(fig_sz * 100 - ydisplay)))
                    if draw_square_number_labels and previous_width > 30:
                        ax.annotate(coord_display, xy=(x + 0.49 * previous_width, y + 0.49 * next_height), fontsize=drawn_number_font_sz)
                else:
                    # We are staying below the horizontal slash.
                    next_center: Tuple[float, float] = (x, y)
                    xdisplay, ydisplay = ax.transData.transform(
                        (x + 0.49 * previous_width, y + next_height + 0.49 * next_height))
                    # coord_display: str = "({}, {})".format(int(xdisplay), int(fig_sz * 100 - ydisplay))
                    coord_display: str = "{}".format(depth)
                    area_details.append((depth, int(xdisplay), int(fig_sz * 100 - ydisplay)))
                    if draw_square_number_labels and previous_width > 30:
                        ax.annotate(coord_display, xy=(x + 0.49 * previous_width, y + next_height + 0.49 * next_height), fontsize=drawn_number_font_sz)
                add_rectangle(current_rectangles, next_center, previous_width, next_height, area_details, depth - 1, prev_dir)


        else:
            # Vertical slash.
            prev_dir = "vertical"
            next_width: float = previous_width / 2
            new_rect: Rectangle = Rectangle(last_rect_center, next_width, previous_height)
            current_rectangles.append(new_rect)

            if depth > 0:
                if random.random() < 0.5:
                    # We are moving to the right of the vertical slash.
                    next_center: Tuple[float, float] = (x + next_width, y)
                    xdisplay, ydisplay = ax.transData.transform((x + 0.49 * next_width, y + previous_height * 0.49))
                    # coord_display: str = "({}, {})".format(int(xdisplay), int(fig_sz * 100 - ydisplay))
                    coord_display: str = "{}".format(depth)
                    area_details.append((depth, int(xdisplay), int(fig_sz * 100 - ydisplay)))
                    if draw_square_number_labels and previous_height > 30:
                        ax.annotate(coord_display, xy=(x + 0.49 * next_width, y + previous_height * 0.49), fontsize=drawn_number_font_sz)
                else:
                    # We are moving to the left of the vertical slash.
                    next_center: Tuple[float, float] = (x, y)
                    xdisplay, ydisplay = ax.transData.transform(
                        (x + next_width + 0.49 * next_width, y + previous_height * 0.49))
                    # coord_display: str = "({}, {})".format(int(xdisplay), int(fig_sz * 100 - ydisplay))
                    coord_display: str = "{}".format(depth)
                    area_details.append((depth, int(xdisplay), int(fig_sz * 100 - ydisplay)))
                    if draw_square_number_labels and previous_height > 30:
                        ax.annotate(coord_display, xy=(x + next_width + 0.49 * next_width, y + previous_height * 0.49), fontsize=drawn_number_font_sz)

                add_rectangle(current_rectangles, next_center, next_width, previous_height, area_details, depth - 1, prev_dir)

    if square_recursion_depth > 0:
        # Store the new rectangles first in a separate list.
        new_rectangles: List[Rectangle] = []

        # Generate 'depth' rectangles.
        # print("> Generating rectangles with recursive strategy on {}.".format(last_center))
        # add_rectangle(new_rectangles, last_center,
        #              previous_width=previous_side,
        #              previous_height=previous_side,
        #              depth=square_recursion_depth)

        area_details: List[Tuple[int, int, int]] = []

        add_rectangle(new_rectangles,
                      last_rect_center=(0, 0),
                      previous_width=fig_sz * 100,
                      previous_height=fig_sz * 100,
                      area_details=area_details,
                      depth=square_recursion_depth)

        # Add to the patch list.
        rectangles.extend(new_rectangles)

    # Output area details.
    if draw_square_number_labels:
        coords_output_path: str = output_image_file_path.replace(".png", ".txt")
        with open(coords_output_path, "w") as coords_file:
            for area_id, x, y in area_details:
                coords_file.write("{}:\t({}, {})\n".format(area_id, x, y))

    # Compute the furthest top-left and bottom-right coordinates that are part of any rectangle.
    # min_x: int = None
    # max_x: int = None
    # min_y: int = None
    # max_y: int = None
    # for r in rectangles:
    #     if min_x is None or r.get_x() < min_x:
    #         min_x = r.get_x()
    #     if min_y is None or r.get_y() < min_y:
    #         min_y = r.get_y()
    #     if max_x is None or r.get_x() + r.get_width() > max_x:
    #         max_x = r.get_x() + r.get_width()
    #     if max_y is None or r.get_y() + r.get_height() > max_y:
    #         max_y = r.get_y() + r.get_height()

    # top_left: Tuple[int, int] = (min_x, min_y)
    # bottom_right: Tuple[int, int] = (max_x, max_y)

    #### DEBUG PRINTS - DELETE LATER OR SWITCH TO -d FLAG
    # pprint.pprint("> Data coordinate top-left:{}\tbottom-right:{}".format(top_left, bottom_right))
    # pprint.pprint("> Data to axis coordinate top-left:{}\tbottom-right:{}".format(ax.transAxes.transform(top_left),
    #                                                                              ax.transAxes.transform(bottom_right)))
    # pprint.pprint("> Axis coordinate to data coordinate (0, 0):{}".format(ax.transAxes.transform((0, 0))))

    # Move and resize all rectangles so the furthest top-left coordinate is now at (0, 0) (axes coordinates).
    # for r in rectangles:
    #     r.set_x(r.get_x() - top_left[0])
    #     r.set_y(r.get_y() - top_left[1])
    #     r.set_width(r.get_width() * (fig_sz * 100) / bottom_right[0])
    #    r.set_height(r.get_height() * (fig_sz * 100) / bottom_right[1])


    # Finally configure the rectangles as a PatchCollection.
    col: PatchCollection = PatchCollection(rectangles, alpha=alpha, edgecolor="black", linewidths=line_width)

    try:
        col.set(array=np.asarray(range(number_of_squares + 1)), cmap=color_map)
    except ValueError:
        print(
            f" '{color_map}' is an invalid colormap, choose a valid one from "
            "https://matplotlib.org/examples/color/colormaps_reference.html"
            " - returning to default 'Blues' colormap"
        )
        col.set(array=np.asarray(range(number_of_squares + 1)), cmap="Blues")

    ax.add_collection(col)
    ax.set_aspect("equal", "box")
    ax.set_xticks([])
    ax.set_yticks([])

    # xmin: int = np.min(xs)
    # ymin: int = np.min(ys)
    # xmax: int = np.max(xs) + fibs[np.argmax(xs)]
    # ymax: int = np.max(ys) + fibs[np.argmax(ys)]
    # ax.set_xlim(xmin, xmax)
    # ax.set_ylim(ymin, ymax)

    # for rect in ax:
    #    pprint.pprint("> rect: {}".format(rect))


    if draw_golden_ratio:
        gr: str = str(fibs[-1] / fibs[-2])
        plt.title(r"$\varphi$ = " + gr)

    plt.tight_layout()
    plt.savefig(output_image_file_path)

    # Create second figure without annotations.
    if draw_square_number_labels:
        for child in ax.get_children():
            if isinstance(child, Annotation):
                child.remove()
                # print("bingo")  # and do something

        no_labels_output_image_file_path: str = output_image_file_path.replace(".png", "NO-LABELS.png")
        plt.savefig(no_labels_output_image_file_path)


if __name__ == "__main__":

    description: str = "Creates a plot for a fibonacci spiral"
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "-n",
        "--number_of_squares",
        type=int,
        help="number of squares in spiral",
        required=True,
    )

    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Plot file name",
        required=False,
        default="plot.png",
    )

    parser.add_argument(
        "--no-square-number-labels",
        dest="draw_square_number_labels",
        help="Remove label showing side length at the center of each square",
        action="store_false",
        default=True,
        required=False,
    )

    parser.add_argument(
        "--no-fibonacci-arc",
        dest="draw_fibonacci_arcs",
        help="Remove arc of fibonacci spiral",
        action="store_false",
        default=True,
        required=False,
    )

    parser.add_argument(
        "--reverse-fib",
        dest="reverse_fib_order",
        help="Reverse list of fibonacci values",
        action="store_true",
        default=False,
        required=False,
    )

    parser.add_argument(
        "--randomize-fib",
        dest="randomize_fib_order",
        help="Randomize list of fibonacci values",
        action="store_true",
        default=False,
        required=False,
    )

    parser.add_argument(
        "-c",
        "--color-map",
        dest="color_map",
        type=str,
        help=(
            """Color map applied to fibonacci squares. See """
            """https://matplotlib.org/stable/tutorials/colors/colormaps.html#sphx-glr-tutorials-colors-colormaps-py"""),
        default="Blues",
        required=False,
    )

    parser.add_argument(
        "-a",
        "--alpha",
        dest="alpha",
        type=float,
        help="Transparency",
        default=0.95,
        required=False,
    )

    parser.add_argument(
        "-l",
        "--line-width",
        dest="line_width",
        type=int,
        help="matplotlib Rectangle line width",
        default=10,
        required=False,
    )

    parser.add_argument(
        "--depth",
        dest="square_recursion_depth",
        type=int,
        help="depth of the random square recursion",
        default=0,
        required=False,
    )

    args: argparse.Namespace = parser.parse_args()

    fib_order: str = "NORMAL"
    suffix: str = ""
    if args.reverse_fib_order and args.randomize_fib_order:
        print("> Only one of '--reverse-fib-order' and '--randomize-fib-order' is allowed. Exiting.")
        sys.exit(1)
    elif args.reverse_fib_order:
        fib_order = "REVERSE"
    elif args.randomize_fib_order:
        fib_order = "RANDOMIZE"



    rand_id: str = str(int(time.time()))
    suffix = "_" + rand_id

    output_path: str = "figures/fib-plot-n{}-l{}-r{}-{}-FIB_ORDER.png".format(args.number_of_squares,
                                                                          args.line_width,
                                                                          args.square_recursion_depth,
                                                                          args.color_map)
    output_path = output_path.replace("ORDER", fib_order + suffix)

    plot_fibonacci_spiral(
        number_of_squares=args.number_of_squares,
        draw_square_number_labels=bool(args.draw_square_number_labels),
        draw_fibonacci_arcs=bool(args.draw_fibonacci_arcs),
        line_width=args.line_width,
        color_map=args.color_map,
        alpha=args.alpha,
        square_recursion_depth=args.square_recursion_depth,
        fib_list_order=fib_order,
        output_image_file_path=output_path
    )
