import cv2
import numpy as np


def detect_ball(frame, debugMode, trackwindow1, trackwindow2, fieldwindow):
    # Convert frame from BGR to GRAY
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if debugMode:
        cv2.imshow("gray", gray)

    # Edge detection using Canny function
    img_edges = cv2.Canny(gray, 50, 190, 3)
    if debugMode:
        cv2.imshow("img_edges", img_edges)

    # Convert to black and white image
    ret, img_thresh = cv2.threshold(img_edges, 254, 255, cv2.THRESH_BINARY)
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

        # Take only the valid circle(s)
        x1, y1, w1, h1 = trackwindow1
        x2, y2, w2, h2 = trackwindow2
        x3, y3, w3, h3 = fieldwindow
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
    # cv2.imshow("contours", img_thresh)
    return centers


def track_tennis_video(
    input_file_name, output_file_name, initial_ball_direction="none"
):
    x1f, y1f = 300, 140
    x2f, y2f = 1600, 1000
    field_window = (x1f, y1f, x2f, y2f)

    file_name = input_file_name
    cap = cv2.VideoCapture(file_name)

    # Take the first frame of the video
    ret, frame = cap.read()

    # Create the background subtractor
    fgbg = cv2.createBackgroundSubtractorMOG2()

    # Initial position and size of players
    x1, y1 = 0, 0
    w1, h1 = 50, 80
    track_window1 = (x1, y1, w1, h1)

    x2, y2 = 0, 0
    w2, h2 = 130, 150
    track_window2 = (x2, y2, w2, h2)

    ballDirection = initial_ball_direction
    changeDirection = False
    # KF = KalmanFilter(0.1, 1, 1, 1, 0.1, 0.1)
    prev_x, prev_y = 0, 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    height, width, _ = frame.shape
    size = (width, height)
    out = cv2.VideoWriter(output_file_name, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    score1, score2 = 0, 0

    showHitCounter = 100
    lastFrame = frame
    while 1:
        ret, frame = cap.read()
        cv2.rectangle(frame, (50, 100), (400, 280), (0, 0, 0), -1)
        cv2.putText(frame, "SCOREBOARD", (60, 140), 0, 1, (255, 255, 255), 2)
        cv2.putText(frame, "Player 1", (60, 190), 0, 1, (0, 255, 0), 2)
        cv2.putText(frame, str(score1), (250, 190), 0, 1, (0, 255, 0), 2)
        cv2.putText(frame, "Player 2", (60, 240), 0, 1, (255, 0, 0), 2)
        cv2.putText(frame, str(score2), (250, 240), 0, 1, (255, 0, 0), 2)
        if ret == True:
            # Remove the background
            blurFrame = frame.copy()

            # Crowd blurring
            ksize = (100, 100)
            blurFrame[:100, :] = cv2.blur(frame[:100, :], ksize)
            blurFrame[:, :350] = cv2.blur(frame[:, :350], ksize)
            blurFrame[:, 1600:] = cv2.blur(frame[:, 1600:], ksize)
            blurFrame[:500, :500] = cv2.blur(frame[:500, :500], ksize)
            blurFrame[:500, 1400:] = cv2.blur(frame[:500, 1400:], ksize)
            blurFrame[:300, :600] = cv2.blur(frame[:300, :600], ksize)
            blurFrame[:300, 1300:] = cv2.blur(frame[:300, 1300:], ksize)

            ksize = (29, 29)
            blurFrame = cv2.GaussianBlur(blurFrame, ksize, 0)

            fgmask = fgbg.apply(blurFrame)

            kernel = np.ones((3, 3), np.uint8)
            fgmask = cv2.erode(fgmask, kernel, iterations=1)

            kernel = np.ones((5, 5), np.uint8)
            fgmask = cv2.dilate(fgmask, kernel, iterations=1)

            kernel = np.ones((5, 5), np.uint8)
            fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

            kernel = np.ones((50, 50), np.uint8)
            fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)

            backtorgb = cv2.cvtColor(fgmask, cv2.COLOR_GRAY2RGB)

            # PLAYERS DETECTION

            # Find contours
            contours, _ = cv2.findContours(
                fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            # Set the accepted minimum & maximum radius of a detected object
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
                    if 100 < centers[i][1] < 425 and 600 < centers[i][0] < 1300:
                        x1, y1 = centers[i][0], centers[i][1]
                        if y1 > 300:
                            w1, h1 = 70, 90
                        else:
                            w1, h1 = 50, 80
                        track_window1 = (x1, y1, w1, h1)
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
                    if 300 < centers[i][1] and 200 < centers[i][0] < 1600:
                        x2, y2 = centers[i][0], centers[i][1]
                        if y2 < 700:
                            w2, h2 = 70, 120
                        else:
                            w2, h2 = 130, 150
                        track_window2 = (x2, y2, w2, h2)
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

            # KALMAN FILTER
            # hit = when distance with the ball is close, and ball changes direction
            centers = detect_ball(
                backtorgb, 0, track_window1, track_window2, field_window
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
                # Predict
                # KF.predict()
                # KF.update(centers[0])
                cv2.putText(
                    frame,
                    "Tennis Ball",
                    (int(centers[0][0] + 15), int(centers[0][1] - 15)),
                    0,
                    0.5,
                    (0, 191, 255),
                    2,
                )
                if current_y > prev_y and 8 < np.abs(current_y - prev_y) < 30:
                    if ballDirection != "down":
                        changeDirection = True
                    ballDirection = "down"
                if current_y < prev_y and 8 < np.abs(current_y - prev_y) < 30:
                    if ballDirection != "up":
                        changeDirection = True
                    ballDirection = "up"
                if changeDirection:
                    showHitCounter = 0
                    if ballDirection == "up":
                        score2 += 1
                    if ballDirection == "down":
                        score1 += 1
                    changeDirection = False
                prev_x, prev_y = current_x, current_y
            if showHitCounter <= int(fps) // 2:
                if ballDirection == "down":
                    cv2.putText(
                        frame,
                        "HIT by player 1!",
                        (width // 4 + 100, height // 2),
                        0,
                        3,
                        (0, 255, 0),
                        10,
                    )
                if ballDirection == "up":
                    cv2.putText(
                        frame,
                        "HIT by player 2!",
                        (width // 4 + 100, height // 2),
                        0,
                        3,
                        (255, 0, 0),
                        10,
                    )
            showHitCounter += 1
            lastFrame = frame
            out.write(frame)
            cv2.imshow("img2", frame)

            k = cv2.waitKey(60) & 0xFF
            if k == 27:
                break
        else:
            break

    frame = lastFrame.copy()
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
    cv2.waitKey(3000)  # Pauses for 3 seconds on the final result

    out.release()
    cv2.destroyAllWindows()
    cap.release()


if __name__ == "__main__":
    input_file_name = "video_cut.mp4"
    output_file_name = "demo_output_" + input_file_name
    initial_ball_direction = "none"
    track_tennis_video(input_file_name, output_file_name, initial_ball_direction)
