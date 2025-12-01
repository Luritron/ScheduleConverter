# import cv2
# import numpy as np
# import pandas as pd
#
#
# def process_schedule_visual(image_path, output_excel):
#     # 1. Завантаження
#     img = cv2.imread(image_path)
#     if img is None:
#         print("Помилка: файл не знайдено.")
#         return
#
#     vis_img = img.copy()  # Картинка для перевірки
#     h_img, w_img = img.shape[:2]
#
#     # 2. Обробка для пошуку ліній
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#                                    cv2.THRESH_BINARY_INV, 51, 5)
#
#     # 3. Знаходимо межі таблиці (ROI)
#     # Шукаємо найбільший прямокутник (це рамка таблиці)
#     contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     table_contour = max(contours, key=cv2.contourArea)
#     x_table, y_table, w_table, h_table = cv2.boundingRect(table_contour)
#
#     # Трохи звужуємо область пошуку (відкидаємо заголовки зверху)
#     # Припускаємо, що "дані" займають нижні 80% таблиці
#     roi_y = y_table + int(h_table * 0.15)
#     roi_h = int(h_table * 0.85)
#
#     # 4. ГОЛОВНИЙ ФОКУС: Знаходимо ПРАВУ грань таблиці
#     # Скануємо вертикальні лінії тільки в правій частині
#     right_part = thresh[:, int(w_img * 0.8):]
#     lines_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, h_img // 20))
#     detected_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, lines_kernel)
#
#     # Знаходимо всі X-координати вертикальних ліній
#     line_contours, _ = cv2.findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     x_lines = []
#     for c in line_contours:
#         lx, ly, lw, lh = cv2.boundingRect(c)
#         if lh > h_img * 0.3:  # Лінія має бути довгою
#             x_lines.append(lx + lw // 2)
#
#     x_lines.sort()
#
#     # Права межа - це остання знайдена вертикальна лінія
#     if not x_lines:
#         print("Не вдалося знайти лінії. Спробуйте краще фото.")
#         return
#
#     grid_right = x_lines[-1]
#
#     # ТЕПЕР МАТЕМАТИКА:
#     # Ми не шукаємо ліву межу. Ми шукаємо ширину колонки.
#     # Знаходимо середню відстань між лініями у правій частині (де години 20-24)
#     # Це найточніший показник ширини стовпця.
#     gaps = []
#     for i in range(len(x_lines) - 1):
#         gap = x_lines[i + 1] - x_lines[i]
#         if w_img * 0.01 < gap < w_img * 0.06:  # Фільтр адекватних відстаней
#             gaps.append(gap)
#
#     if not gaps:
#         col_width = w_table / 27  # Аварійний варіант
#     else:
#         # Беремо медіану, щоб відкинути помилки
#         col_width = np.median(gaps)
#
#     print(f"Розрахована ширина колонки: {col_width:.2f} пікселів")
#
#     # 5. Будуємо сітку рядків
#     # Розбиваємо висоту таблиці (без шапки) на 12 рівних частин
#     # Тому що 6 черг * 2 підчерги = 12 рядків. Це стандарт.
#     row_height = roi_h / 12
#
#     schedule_result = {}
#     # Порядок черг: 1.1, 1.2, 2.1... -> 1, 1, 2, 2...
#     queue_map = [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6]
#
#     for row_idx in range(12):
#         # Координати рядка
#         y1 = int(roi_y + row_idx * row_height)
#         y2 = int(roi_y + (row_idx + 1) * row_height)
#
#         q_num = queue_map[row_idx]
#         if q_num not in schedule_result:
#             schedule_result[q_num] = [""] * 24
#
#         # Проходимо по 24 годинам СПРАВА НАЛІВО
#         for hour_inv in range(24):
#             hour_real = 23 - hour_inv
#
#             # Координати колонки (відштовхуємось від ПРАВОГО краю)
#             x2 = int(grid_right - (hour_inv * col_width))
#             x1 = int(grid_right - ((hour_inv + 1) * col_width))
#
#             # Малюємо сітку на дебаг-фото (зелений - межі)
#             cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 1)
#
#             # Зона аналізу (центр клітинки)
#             margin_x = int(col_width * 0.35)  # Великий відступ, щоб не чіпляти сусідів
#             margin_y = int(row_height * 0.35)
#
#             roi_x1, roi_x2 = x1 + margin_x, x2 - margin_x
#             roi_y1, roi_y2 = y1 + margin_y, y2 - margin_y
#
#             # Перевірка виходу за межі
#             if roi_x1 >= roi_x2 or roi_y1 >= roi_y2: continue
#
#             # Вирізаємо шматочок
#             cell_roi = img[roi_y1:roi_y2, roi_x1:roi_x2]
#
#             # Шукаємо синій
#             hsv = cv2.cvtColor(cell_roi, cv2.COLOR_BGR2HSV)
#             lower_blue = np.array([90, 50, 50])
#             upper_blue = np.array([170, 255, 255])
#             mask = cv2.inRange(hsv, lower_blue, upper_blue)
#
#             blue_cnt = cv2.countNonZero(mask)
#             total_px = (roi_x2 - roi_x1) * (roi_y2 - roi_y1)
#
#             if total_px > 0 and (blue_cnt / total_px) > 0.3:  # Поріг 30% синього
#                 schedule_result[q_num][hour_real] = "+"
#                 # Малюємо червоний кружок, якщо знайшли відключення
#                 cv2.circle(vis_img, (x1 + int(col_width / 2), y1 + int(row_height / 2)),
#                            5, (0, 0, 255), -1)
#
#     # 6. Зберігаємо результати
#
#     # 6.1 Зображення з сіткою (ОБОВ'ЯЗКОВО ПЕРЕВІРТЕ ЙОГО)
#     cv2.imwrite("debug_grid.jpg", vis_img)
#     print("Створено файл 'debug_grid.jpg'. Відкрийте його та перевірте, чи рівно лежить сітка.")
#
#     # 6.2 Excel
#     final_data = []
#     headers = [f"{h:02d}-{h + 1:02d}" for h in range(24)]
#
#     for q in sorted(schedule_result.keys()):
#         row = {"Черга": q}
#         for h_idx, val in enumerate(schedule_result[q]):
#             row[headers[h_idx]] = val
#         final_data.append(row)
#
#     df = pd.DataFrame(final_data)
#     df.to_excel(output_excel, index=False)
#     print(f"Таблиця збережена як: {output_excel}")
#
#
# if __name__ == "__main__":
#     process_schedule_visual("schedule.jpg", "Schedule_table.xlsx")

import cv2
import numpy as np
import pandas as pd


def process_schedule_desire_match(image_path, output_excel):
    # 1. Завантаження
    img = cv2.imread(image_path)
    if img is None:
        print("Помилка: файл не знайдено.")
        return

    vis_img = img.copy()
    h_img, w_img = img.shape[:2]

    # 2. Попередня обробка
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 51, 5)

    # 3. ПОШУК ВЕРТИКАЛЬНИХ ЛІНІЙ (ВСІХ)
    # Використовуємо високе ядро, щоб ігнорувати текст, але ловити лінії таблиці
    kernel_h = int(h_img * 0.05)
    ver_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_h))
    detected_ver = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, ver_kernel)

    col_sums = np.sum(detected_ver, axis=0)
    # Поріг чутливості
    x_indices = np.where(col_sums > np.max(col_sums) * 0.2)[0]

    lines_x = []
    if len(x_indices) > 0:
        current_group = [x_indices[0]]
        for x in x_indices[1:]:
            if x - current_group[-1] < 10:
                current_group.append(x)
            else:
                lines_x.append(int(np.mean(current_group)))
                current_group = [x]
        lines_x.append(int(np.mean(current_group)))

    if not lines_x:
        print("Лінії не знайдено.")
        return

    # 4. ПОБУДОВА ЗЕЛЕНОЇ СІТКИ (АЛГОРИТМ ПРИЛИПАННЯ)

    # Крок А: Правий край (Right Anchor)
    # Це остання лінія на фото
    right_anchor = lines_x[-1]

    # Крок Б: Вираховуємо "Ідеальний крок" (Average Step)
    # Беремо тільки праву частину фото, де гарантовано чиста сітка
    clean_lines = [x for x in lines_x if x > w_img * 0.6]
    gaps = np.diff(clean_lines)
    valid_gaps = [g for g in gaps if w_img * 0.015 < g < w_img * 0.08]

    if not valid_gaps:
        step_px = w_img / 30  # Fallback
    else:
        step_px = np.median(valid_gaps)

    # Крок В: Генерація координат стовпців (Right -> Left)
    # Ми формуємо список з 25 ліній (межі 24 колонок)
    grid_lines = [right_anchor]

    current_x = right_anchor

    for i in range(24):
        # Розрахункова позиція наступної лінії зліва
        target_x = current_x - step_px

        # Спроба знайти РЕАЛЬНУ лінію поруч з target_x (Snap to Grid)
        # Шукаємо в радіусі 30% від кроку
        search_radius = step_px * 0.3
        best_match = None
        min_dist = float('inf')

        for lx in lines_x:
            dist = abs(lx - target_x)
            if dist < min_dist and dist < search_radius:
                min_dist = dist
                best_match = lx

        if best_match is not None:
            # Знайшли реальну лінію! Прилипаємо до неї.
            current_x = best_match
        else:
            # Лінія стерта або не знайдена, використовуємо математику
            current_x = int(target_x)

        grid_lines.append(current_x)

    # Сортуємо, бо ми йшли задом наперед
    grid_lines.sort()

    # Ліва межа - це перша лінія в нашому списку
    left_anchor = grid_lines[0]

    # 5. ПОШУК ГОРИЗОНТАЛЬНИХ МЕЖ (РЯДКІВ)
    hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (int(w_img * 0.1), 1))
    detected_hor = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, hor_kernel)
    row_sums = np.sum(detected_hor, axis=1)
    y_indices = np.where(row_sums > np.max(row_sums) * 0.2)[0]

    lines_y = []
    if len(y_indices) > 0:
        current_group = [y_indices[0]]
        for y in y_indices[1:]:
            if y - current_group[-1] < 10:
                current_group.append(y)
            else:
                lines_y.append(int(np.mean(current_group)))
                current_group = [y]
        lines_y.append(int(np.mean(current_group)))

    # Нам треба знайти верхню і нижню межу таблиці даних
    # Зазвичай це останні 13 ліній (12 рядків + закриваюча)
    if len(lines_y) >= 13:
        # Припускаємо, що це нижня частина таблиці
        table_top = lines_y[-13]
        table_bottom = lines_y[-1]
    else:
        table_bottom = lines_y[-1] if lines_y else h_img
        # Якщо ліній мало, пробуємо вгадати висоту рядка
        # Припускаємо, що рядок трохи вищий за ширину стовпця (або схожий)
        est_height = (table_bottom - lines_y[0]) if lines_y else h_img * 0.8
        table_top = table_bottom - est_height  # Fallback

    row_height = (table_bottom - table_top) / 12

    # === ВІЗУАЛІЗАЦІЯ "DESIRE" (Червоні рамки, Зелена сітка) ===

    # 1. Червона рамка (Межі даних)
    cv2.rectangle(vis_img, (left_anchor, table_top), (right_anchor, table_bottom), (0, 0, 255), 3)

    # 2. Зелена сітка (Вертикальні лінії)
    for x in grid_lines:
        cv2.line(vis_img, (x, table_top), (x, table_bottom), (0, 255, 0), 2)

    # 3. Зелена сітка (Горизонтальні лінії)
    for i in range(13):
        y = int(table_top + i * row_height)
        cv2.line(vis_img, (left_anchor, y), (right_anchor, y), (0, 255, 0), 2)

    # 6. СКАНУВАННЯ ТА ЕКСПОРТ
    schedule_result = {}
    queue_map = [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6]

    for row_idx in range(12):
        y1 = int(table_top + row_idx * row_height)
        y2 = int(table_top + (row_idx + 1) * row_height)
        y_center = int((y1 + y2) / 2)

        q_num = queue_map[row_idx]
        if q_num not in schedule_result:
            schedule_result[q_num] = [""] * 24

        for col_idx in range(24):
            # Беремо координати з нашого "прилипшого" списку
            x1 = grid_lines[col_idx]
            x2 = grid_lines[col_idx + 1]
            x_center = int((x1 + x2) / 2)

            # ROI (Центр)
            margin_x = int((x2 - x1) * 0.25)
            margin_y = int(row_height * 0.25)

            roi_x1 = x1 + margin_x
            roi_x2 = x2 - margin_x
            roi_y1 = y1 + margin_y
            roi_y2 = y2 - margin_y

            if roi_x1 >= roi_x2 or roi_y1 >= roi_y2: continue

            roi = img[roi_y1:roi_y2, roi_x1:roi_x2]
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

            # ФІЛЬТР: Saturation > 60 (Ігноруємо текст)
            lower_blue = np.array([90, 60, 50])
            upper_blue = np.array([170, 255, 255])
            mask = cv2.inRange(hsv, lower_blue, upper_blue)

            count = cv2.countNonZero(mask)
            total = (roi_x2 - roi_x1) * (roi_y2 - roi_y1)

            if total > 0 and (count / total) > 0.15:
                schedule_result[q_num][col_idx] = "+"
                # 3. Червоні цятки (Desire style)
                cv2.circle(vis_img, (x_center, y_center), 5, (0, 0, 255), -1)

    # Збереження зображення
    cv2.imwrite("debug_grid.jpg", vis_img)
    print("Згенеровано 'debug_grid.jpg'. Перевірте відповідність з 'desire.png'.")

    # Експорт в Excel
    final_data = []
    headers = [f"{h:02d}-{h + 1:02d}" for h in range(24)]

    for q in sorted(schedule_result.keys()):
        row = {"Черга": q}
        for h_idx, val in enumerate(schedule_result[q]):
            row[headers[h_idx]] = val
        final_data.append(row)

    df = pd.DataFrame(final_data)
    df.to_excel(output_excel, index=False)
    print(f"Таблиця готова: {output_excel}")


if __name__ == "__main__":
    process_schedule_desire_match("schedule.jpg", "Schedule_excel.xlsx")