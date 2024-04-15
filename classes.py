import easyocr
import utils
import cv2 as cv

class Data_table:
    def __init__(self, image, contours) -> None:
        self.image = image
        self.contours = self.sort_contours(contours)
        self.img_cells = self.crop_img(self.image, self.contours)
        self.data_table = []
        
        for i in range(len(self.img_cells)):
            col = []
            for j in range(len(self.img_cells[i])):
                utils.mkdir("bin/cells")
                cv.imwrite(('bin/cells/' + 'cell' + ' ' + str(i) + ' ' + str(j) + '.jpg'), self.img_cells[i][j])
                col.append(self.text_recording(self.img_cells[i][j]))
            self.data_table.append(col)

    # сортировка всех найденных контуров таблицы
    def sort_contours(self, contours):
        sorted_contours = []
        item_list = []
        contours.sort(key=lambda x: x[1])
        prev_cont = contours[0]
        for cont in contours:
            if prev_cont[1]-50 < cont[1] < prev_cont[1]+50:
                item_list.append(cont)
            else:
                sorted_contours.append(item_list)
                item_list = []
                item_list.append(cont)
            prev_cont = cont
        sorted_contours.append(item_list)
        for cont in sorted_contours:
            cont.sort()
        return sorted_contours
    
    # Обрезка найденной таблицы на более маленькие изображений. Каждое изображение равно одной ячейке таблицы 
    def crop_img(self, image,coordinates):
        crop_images = []
        row = []
        for coord in coordinates:
            for item in coord:
                x,y,w,h = item
                row.append(image[y:y+h, x:x+w])
            crop_images.append(row)
            row = []
        return crop_images
    
    
    # Статический метод распознавания таблицы. Имеет два типа для распознавнаия
    # String - распозование всего текста на изображении (Используется для ячеек)
    # StringList - Распознавание текста на изображении с координатами углов (Используется для нахождения ячеек)
    @staticmethod
    def text_recording(file, type='String'):
        reader = easyocr.Reader(["ru", "en"], gpu=True)
        if type == 'StringList':
            return reader.readtext(file, detail=1)
        elif type == 'String':
            return ''.join(reader.readtext(file, detail=0, paragraph=True))



class Table:
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.joints = None

    def __str__(self):
        return "(x: %d, y: %d, w: %d, h: %d)" % (self.x, self.x + self.w, self.y, self.y + self.h)
    
    # Сохраняет координаты соединений таблицы.
    # Предполагается, что соединения n-мерного массива отсортированы в порядке возрастания.
    def set_joints(self, joints):
        if self.joints != None:
            raise ValueError("Invalid setting of table joints array.")

        self.joints = []
        row_y = joints[0][1]
        row = []
        for i in range(len(joints)):
            if i == len(joints) - 1:
                row.append(joints[i])
                self.joints.append(row)
                break

            row.append(joints[i])

            # Если при следующем добавлении получается новая координата y,
            # Начинаем новую строчку.
            if joints[i + 1][1] != row_y:
                self.joints.append(row)
                row_y = joints[i + 1][1]
                row = []

    # печать координат соединений таблицы
    def print_joints(self):
        if self.joints == None:
            print("Joint coordinates not found.")
            return

        print("[")
        for row in self.joints:
            print("\t" + str(row))
        print("]")

    # Находит границы записей таблицы на изображении по используемым координатам соединений таблицы.
    def get_table_entries(self):
        if self.joints == None:
            print("Joint coordinates not found.")
            return

        entry_coords = []
        for i in range(0, len(self.joints) - 1):
            entry_coords.append(self.get_entry_bounds_in_row(self.joints[i], self.joints[i + 1]))

        return entry_coords

    # Нахождение границы записей таблицы в каждой строке на основе заданных наборов соединений
    def get_entry_bounds_in_row(self, joints_A, joints_B):
        row_entries = []

        # Поскольку наборы соединений могут иметь разное количество точек, для нахождения границ мы выбираем набор с меньшим количеством точек
        if len(joints_A) <= len(joints_B):
            defining_bounds = joints_A
            helper_bounds = joints_B
        else:
            defining_bounds = joints_B
            helper_bounds = joints_A

        for i in range(0, len(defining_bounds) - 1):
            x = defining_bounds[i][0]
            y = defining_bounds[i][1]
            w = defining_bounds[i + 1][0] - x # (i + 1)-я координата helper_bounds не может быть правым нижним углом.
            h = helper_bounds[0][1] - y # helper_bounds имеет одинаковую координату Y для всех своих элементов.

            # Если вычисленная высота меньше 0, делаем высоту положительной и используйте координату Y строки выше для границ.
            if h < 0:
                h = -h
                y = y - h

            row_entries.append([x, y, w, h])

        return row_entries
