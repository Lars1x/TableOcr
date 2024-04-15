import classes
import cv2 as cv
import subprocess as s
import os
import numpy as np
import xlsxwriter

# Применение морфологических операций

def isolate_lines(src, structuring_element):
	cv.erode(src, structuring_element, src, (-1, -1)) # Уменьшение белых пятен
	cv.dilate(src, structuring_element, src, (-1, -1)) # Увеличение белых пятен

# Проверка, является ли область внутри контура таблицей
# Если это таблица, возвращает ограничивающий прямоугольник.
# и соединения таблицы


MIN_TABLE_AREA = 50 # минимальная область таблицы
EPSILON = 3 # значение эпсилон для аппроксимации контура


def verify_table(contour, intersections):
    area = cv.contourArea(contour)

    if (area < MIN_TABLE_AREA):
        return (None, None)

    # ApprolyDP аппроксимирует полигональную кривую с заданной точностью
    curve = cv.approxPolyDP(contour, EPSILON, True)

    # boundingRect вычисляет ограничивающий прямоугольник набора точек (например, кривая)
    rect = cv.boundingRect(curve) # format of each rect: x, y, w, h

    # Находит количество соедиений в каждой интересующей области (ROI)
    # Формат находится в порядке строк и столбцов (поскольку для поиска используются массивы numpy).
    possible_table_region = intersections[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]]
    (possible_table_joints, _) = cv.findContours(possible_table_region, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)

    # Определение количество соединений таблицы в изображении.
    # Если меньше 5 соединений таблицы, то изображение, скорее всего, не является таблицей.
    
    if len(possible_table_joints) < 5:
        return (None, None)

    return rect, possible_table_joints


# Получение контуров таблицы
def get_contours(thresh_value):
    kernel = np.ones((2,2),np.uint8)    
    dilated_value = cv.dilate(thresh_value,kernel,iterations = 1)

    contours, hierarchy = cv.findContours(dilated_value,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
    coordinates = []
    for contour in contours:
        x,y,w,h = cv.boundingRect(contour)
        if h> 50 and w>50 and h*w<thresh_value.shape[0]*thresh_value.shape[1]:  
            coordinates.append((x,y,w,h))
    return coordinates

# Вывод данных в формат excel (Table_data является вложенным списком)
def excel_export(Table_data):
    mkdir("excel/")
    workbook = xlsxwriter.Workbook('excel/tables.xlsx')
    worksheet = workbook.add_worksheet()
    for i in range(len(Table_data)):
        for j in range(len(Table_data[i])):
            worksheet.write(i, j, Table_data[i][j])
    workbook.close()

### Создание директории (для удобства)

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

### Поиск таблиц на заданном изображении и сохранение их в отдельные файлы
def search_for_tables(image):
        NUM_CHANNELS = 3
        if len(image.shape) == NUM_CHANNELS:
            grayscale = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

        MAX_THRESHOLD_VALUE = 255
        BLOCK_SIZE = 15
        THRESHOLD_CONSTANT = 0

        # фильтрация изображения
        filtered = cv.adaptiveThreshold(~grayscale, MAX_THRESHOLD_VALUE, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, BLOCK_SIZE, THRESHOLD_CONSTANT)
 
        SCALE = 15

        horizontal = filtered.copy()
        vertical = filtered.copy()

        horizontal_size = int(horizontal.shape[1] / SCALE)
        horizontal_structure = cv.getStructuringElement(cv.MORPH_RECT, (horizontal_size, 1))
        isolate_lines(horizontal, horizontal_structure)

        vertical_size = int(vertical.shape[0] / SCALE)
        vertical_structure = cv.getStructuringElement(cv.MORPH_RECT, (1, vertical_size))
        isolate_lines(vertical, vertical_structure)

        mask = horizontal + vertical
        (contours, _) = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        intersections = cv.bitwise_and(horizontal, vertical)

        tables = [] 
        for i in range(len(contours)):
            (rect, table_joints) = verify_table(contours[i], intersections)
            if rect == None or table_joints == None:
                continue
            
            table = classes.Table(rect[0], rect[1], rect[2], rect[3])

            joint_coords = []
            for i in range(len(table_joints)):
                joint_coords.append(table_joints[i][0][0])
            joint_coords = np.asarray(joint_coords)

            sorted_indices = np.lexsort((joint_coords[:, 0], joint_coords[:, 1]))
            joint_coords = joint_coords[sorted_indices]

            table.set_joints(joint_coords)

            tables.append(table)

            cv.rectangle(image, (table.x, table.y), (table.x + table.w, table.y + table.h), (0, 255, 0), 1, 8, 0)
        out = "bin/"
        table_name = "table.jpg"
        mult = 3
        mkdir(out)
        mkdir("bin/table/")
        seached_tables = []
        k = len(tables)-1
        for table in tables:
            table_name = "table"+str(k)+".jpg"

            table_roi = image[table.y:table.y + table.h, table.x:table.x + table.w]
            table_roi = cv.resize(table_roi, (table.w * mult, table.h * mult))

            seached_tables.append(table_roi)

            cv.imwrite(out + table_name, table_roi)
            k=k-1
        return seached_tables
    
# Удаление всего текста из области найденой таблицы для выявления всех существующих ячеек,
# для избежания долонительных ошибок сканирования table - изображение самой таблицы

def remove_text(table):
    table = cv.cvtColor(table, cv.COLOR_BGR2GRAY)
    thresh = 200
    img_binary = cv.threshold(table, thresh, 255, cv.THRESH_BINARY_INV)[1]
    cv.imwrite('black-and-white.png',img_binary)

    fields = classes.Data_table.text_recording(file=img_binary, type='StringList')

    for field in fields:
        heigh = field[0][2][1] - field[0][0][1]
        buff = heigh * 0.15
        field[0][0][1] = int(field[0][0][1] + buff/2)
        field[0][2][1] = int(field[0][2][1] - buff/2)
        cv.rectangle(img_binary,
                    pt1=(int(field[0][0][0]), int(field[0][0][1])),
                    pt2=(int(field[0][2][0]), int(field[0][2][1])),
                    color=(0,0,0), thickness=-1)
        
    thickness = int(table.shape[1] * 0.015)
    cv.rectangle(img_binary,(0,0),(table.shape[1],table.shape[0]),(255,255,255), thickness=thickness)
    cv.imwrite("resultIMg.png", img_binary)
    return img_binary