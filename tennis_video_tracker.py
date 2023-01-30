import cv2
import numpy as np


def detect_ball(frame, debugMode, track_window_1, track_window_2, field_window):
    # Convert frame from BGR to GRAY
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if debugMode:
        cv2.imshow("gray", gray)

    # Edge detection using Canny function
    img_edges = cv2.Canny(gray, 50, 190, 3)
    if debugMode:
        cv2.imshow("img_edges", img_edges)

    # Convert to black and white image
    _, img_thresh = cv2.threshold(img_edges, 254, 255, cv2.THRESH_BINARY)
    if debugMode:
        cv2.imshow("img_thresh", img_thresh)

    # Find contours
    contours, _ = cv2.findContours(
        img_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Set the accepted minimum & maximum radius of a detected object
    min_radius_thresh = 5
    max_radius_thresh = 15

    centers = []
    for c in contours:
        # ref: https://docs.opencv.org/trunk/dd/d49/tutorial_py_contour_features.html
        (x, y), radius = cv2.minEnclosingCircle(c)
        radius = int(radius)

        # Take only the valid circle(s) within the tennis field and not in the players boxes
        x1, y1, w1, h1 = track_window_1
        x2, y2, w2, h2 = track_window_2
        x3, y3, w3, h3 = field_window
        err_val = 10
        if (x1 - w1 - err_val < x < x1 + w1 + err_val) and (
            y1 - h1 - err_val < y < y1 + h1 + err_val
        ):
            continue
        if (x2 - w2 - err_val < x < x2 + w2 + err_val) and (
            y2 - h2 - err_val < y < y2 + h2 + err_val
        ):
            continue
        if (x3 < x < w3) and (y3 < y < h3):
            if (radius > min_radius_thresh) and (radius < max_radius_thresh):
                centers.append(np.array([[x], [y]]))
    return centers


def track_tennis_video(
    input_file_name, output_file_name, initial_ball_direction="none"
):
    # Initialize field coordinates
    x1f, y1f = 300, 140
    x2f, y2f = 1600, 1000
    field_window = (x1f, y1f, x2f, y2f)

    # Opening the input file
    file_name = input_file_name
    cap = cv2.VideoCapture(file_name)

    # Take the first frame of the video
    ret, frame = cap.read()

    # Create the background subtractor
    fgbg = cv2.createBackgroundSubtractorMOG2()

    # Initial position and size of players
    x1, y1 = 0, 0
    w1, h1 = 50, 80
    track_window_1 = (x1, y1, w1, h1)

    x2, y2 = 0, 0
    w2, h2 = 130, 150
    track_window_2 = (x2, y2, w2, h2)

    # Initialize variables for ball trackign
    ball_direction = initial_ball_direction
    change_direction = False
    prev_x, prev_y = 0, 0

    # Initialize output file
    fps = cap.get(cv2.CAP_PROP_FPS)
    height, width, _ = frame.shape
    size = (width, height)
    out = cv2.VideoWriter(output_file_name, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)

    # Track players score
    score1, score2 = 0, 0

    # Show if a player hits
    show_hit_counter = 100
    last_frame = frame
    while 1:
        ret, frame = cap.read()

        # Show the scoreboard
        cv2.rectangle(frame, (50, 100), (400, 280), (0, 0, 0), -1)
        cv2.putText(frame, "SCOREBOARD", (60, 140), 0, 1, (255, 255, 255), 2)
        cv2.putText(frame, "Player 1", (60, 190), 0, 1, (0, 255, 0), 2)
        cv2.putText(frame, str(score1), (250, 190), 0, 1, (0, 255, 0), 2)
        cv2.putText(frame, "Player 2", (60, 240), 0, 1, (255, 0, 0), 2)
        cv2.putText(frame, str(score2), (250, 240), 0, 1, (255, 0, 0), 2)
        if ret == True:

            # BACKGROUND REMOVAL

            blur_frame = frame.copy()

            # Crowd blurring to the out-field sections
            ksize = (100, 100)
            blur_frame[:100, :] = cv2.blur(frame[:100, :], ksize)
            blur_frame[:, :350] = cv2.blur(frame[:, :350], ksize)
            blur_frame[:, 1600:] = cv2.blur(frame[:, 1600:], ksize)
            blur_frame[:500, :500] = cv2.blur(frame[:500, :500], ksize)
            blur_frame[:500, 1400:] = cv2.blur(frame[:500, 1400:], ksize)
            blur_frame[:300, :600] = cv2.blur(frame[:300, :600], ksize)
            blur_frame[:300, 1300:] = cv2.blur(frame[:300, 1300:], ksize)

            # Gaussian blurring
            ksize = (29, 29)
            blur_frame = cv2.GaussianBlur(blur_frame, ksize, 0)

            # Background removal with MOG2
            fgmask = fgbg.apply(blur_frame)

            # Morphology operations: Erotion -> Dilation -> Opening -> Closing
            kernel = np.ones((3, 3), np.uint8)
            fgmask = cv2.erode(fgmask, kernel, iterations=1)

            kernel = np.ones((5, 5), np.uint8)
            fgmask = cv2.dilate(fgmask, kernel, iterations=1)

            kernel = np.ones((5, 5), np.uint8)
            fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

            kernel = np.ones((50, 50), np.uint8)
            fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)

            # PLAYERS DETECTION

            # Find contours
            contours, _ = cv2.findContours(
                fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            # Set the accepted minimum & maximum radius of a player
            min_radius_thresh = 30
            max_radius_thresh = 3000
            centers = []
            for c in contours:
                (x, y), radius = cv2.minEnclosingCircle(c)
                radius = int(radius)
                if (radius > min_radius_thresh) and (radius < max_radius_thresh):
                    centers.append(np.array([[x], [y]]))

            if len(centers) > 0:
                for i in range(len(centers)):
                    # Draw Player 1 from upper field
                    if 100 < centers[i][1] < 425 and 600 < centers[i][0] < 1300:
                        x1, y1 = centers[i][0], centers[i][1]
                        if y1 > 300:
                            w1, h1 = 70, 90
                        else:
                            w1, h1 = 50, 80
                        track_window_1 = (x1, y1, w1, h1)
                        cv2.rectangle(
                            frame,
                            (int(centers[i][0]) - w1, int(centers[i][1]) - h1),
                            (int(centers[i][0]) + w1, int(centers[i][1]) + h1),
                            (0, 255, 0),
                            2,
                        )
                        cv2.putText(
                            frame,
                            "Player 1",
                            (int(centers[i][0] + h1 + 5), int(centers[i][1] - 15)),
                            0,
                            0.9,
                            (0, 255, 0),
                            2,
                        )
                        break
                for i in range(len(centers)):
                    # Draw Player 2 from lower field
                    if 300 < centers[i][1] and 200 < centers[i][0] < 1600:
                        x2, y2 = centers[i][0], centers[i][1]
                        if y2 < 700:
                            w2, h2 = 70, 120
                        else:
                            w2, h2 = 130, 150
                        track_window_2 = (x2, y2, w2, h2)
                        cv2.rectangle(
                            frame,
                            (int(centers[i][0]) - w2, int(centers[i][1]) - h2),
                            (int(centers[i][0]) + w2, int(centers[i][1]) + h2),
                            (255, 0, 0),
                            2,
                        )
                        cv2.putText(
                            frame,
                            "Player 2",
                            (int(centers[i][0] + w2 + 5), int(centers[i][1])),
                            0,
                            0.9,
                            (255, 0, 0),
                            2,
                        )
                        break

            # BALL DETECTION
            backtorgb = cv2.cvtColor(fgmask, cv2.COLOR_GRAY2RGB)
            centers = detect_ball(
                backtorgb, 0, track_window_1, track_window_2, field_window
            )
            if len(centers) > 0:
                current_x = int(centers[0][0])
                current_y = int(centers[0][1])
                cv2.circle(
                    frame,
                    (int(centers[0][0]), int(centers[0][1])),
                    10,
                    (0, 191, 255),
                    2,
                )
                cv2.putText(
                    frame,
                    "Tennis Ball",
                    (int(centers[0][0] + 15), int(centers[0][1] - 15)),
                    0,
                    0.5,
                    (0, 191, 255),
                    2,
                )

                # HIT DETECTION
                # hit = when distance with the ball is close, and ball changes direction
                if current_y > prev_y and 8 < np.abs(current_y - prev_y) < 200:
                    if ball_direction != "down":
                        change_direction = True
                    ball_direction = "down"
                if current_y < prev_y and 8 < np.abs(current_y - prev_y) < 200:
                    if ball_direction != "up":
                        change_direction = True
                    ball_direction = "up"
                if change_direction:
                    show_hit_counter = 0
                    if ball_direction == "up":
                        score2 += 1
                    if ball_direction == "down":
                        score1 += 1
                    change_direction = False
                prev_x, prev_y = current_x, current_y

            # Show the hit action information on the middle of the screen
            if show_hit_counter <= int(fps) // 2:
                if ball_direction == "down":
                    cv2.putText(
                        frame,
                        "HIT by player 1!",
                        (width // 4 + 100, height // 2),
                        0,
                        3,
                        (0, 255, 0),
                        10,
                    )
                if ball_direction == "up":
                    cv2.putText(
                        frame,
                        "HIT by player 2!",
                        (width // 4 + 100, height // 2),
                        0,
                        3,
                        (255, 0, 0),
                        10,
                    )
            show_hit_counter += 1
            last_frame = frame

            # Output writing
            out.write(frame)
            cv2.imshow("Current frame", frame)

            k = cv2.waitKey(60) & 0xFF
            if k == 27:
                break
        else:
            break

    # SHOWING FINAL RESULTS
    frame = last_frame.copy()
    cv2.rectangle(
        frame,
        (width // 4 - 100, height // 2 - 100),
        (width // 4 + 1000, height // 2 + 50),
        (0, 0, 0),
        -1,
    )
    if score1 > score2:
        cv2.putText(
            frame, "WINNER : Player 1", (width // 4, height // 2), 0, 3, (0, 255, 0), 10
        )
    if score1 < score2:
        cv2.putText(
            frame, "WINNER : Player 2", (width // 4, height // 2), 0, 3, (255, 0, 0), 10
        )
    if score1 == score2:
        cv2.putText(
            frame,
            "IT'S A DRAW!",
            (width // 4 + 180, height // 2),
            0,
            3,
            (255, 255, 255),
            10,
        )
    cv2.imshow("img2", frame)
    for i in range(int(fps) * 3):
        out.write(frame)
    # Pauses for 3 seconds on the final result
    cv2.waitKey(3000)

    out.release()
    cv2.destroyAllWindows()
    cap.release()


# INITIALIZATION
if __name__ == "__main__":
    input_file_name = "video_cut.mp4"
    output_file_name = "demo_output_" + input_file_name
    initial_ball_direction = "none"
    track_tennis_video(input_file_name, output_file_name, initial_ball_direction)
