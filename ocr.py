import easyocr
import numpy as np
import cv2 as cv
from PIL import Image
import utils
from table import Table
import time
import os

class Data_table:
    def __init__(self, image, contours) -> None:
        self.image = image
        self.contours = self.sort_contours(contours)
        self.img_cells = self.crop_img(self.image, self.contours)
        self.data_table = []
        
        for i in range(len(self.img_cells)):
            col = []
            for j in range(len(self.img_cells[i])):
                col.append(self.text_recording(self.img_cells[i][j]))
            self.data_table.append(col)

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
    
    @staticmethod
    def text_recording(file, type='String'):
        reader = easyocr.Reader(["ru", "en"])
        if type == 'StringList':
            return reader.readtext(file, detail=1)
        elif type == 'String':
            return ''.join(reader.readtext(file, detail=0, paragraph=True))

def remove_text(table):
    table = cv.cvtColor(table, cv.COLOR_BGR2GRAY)
    thresh = 200
    img_binary = cv.threshold(table, thresh, 255, cv.THRESH_BINARY_INV)[1]
    cv.imwrite('black-and-white.png',img_binary)

    fields = Data_table.text_recording(file=img_binary, type='StringList')

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

def get_data_table(table, couters):
    return Data_table(table, couters).data_table

def excel_export(Table_data):
    utils.mkdir("excel/")
    workbook = xlsxwriter.Workbook('excel/tables.xlsx')
    worksheet = workbook.add_worksheet()
    for i in range(len(Table_data)):
        for j in range(len(Table_data[i])):
            worksheet.write(i, j, Table_data[i][j])
    workbook.close()

def search_for_tables(image):
        NUM_CHANNELS = 3
        if len(image.shape) == NUM_CHANNELS:
            grayscale = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

        MAX_THRESHOLD_VALUE = 255
        BLOCK_SIZE = 15
        THRESHOLD_CONSTANT = 0

        # Filter image
        filtered = cv.adaptiveThreshold(~grayscale, MAX_THRESHOLD_VALUE, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, BLOCK_SIZE, THRESHOLD_CONSTANT)
 
        SCALE = 15

        horizontal = filtered.copy()
        vertical = filtered.copy()

        horizontal_size = int(horizontal.shape[1] / SCALE)
        horizontal_structure = cv.getStructuringElement(cv.MORPH_RECT, (horizontal_size, 1))
        utils.isolate_lines(horizontal, horizontal_structure)

        vertical_size = int(vertical.shape[0] / SCALE)
        vertical_structure = cv.getStructuringElement(cv.MORPH_RECT, (1, vertical_size))
        utils.isolate_lines(vertical, vertical_structure)

        mask = horizontal + vertical
        (contours, _) = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        intersections = cv.bitwise_and(horizontal, vertical)

        tables = [] 
        for i in range(len(contours)):
            (rect, table_joints) = utils.verify_table(contours[i], intersections)
            if rect == None or table_joints == None:
                continue

            table = Table(rect[0], rect[1], rect[2], rect[3])

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
        utils.mkdir(out)
        utils.mkdir("bin/table/")
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

def main():
    start = time.time()

    image_path =  r'data.png'

    ext_img = Image.open(image_path)
    utils.mkdir("data/")
    ext_img.save("data/target.png", "PNG")
    image = cv.imread("data/target.png")

    tables = search_for_tables(image)

    tabular_data = []

    for table in tables:
        contours = get_contours(remove_text(table))
        tabular_data = Data_table(table, contours).data_table

    print(tabular_data)
    end = time.time() - start
    print(end)
    os.system("pause")
        




if __name__ == "__main__":
    main()


