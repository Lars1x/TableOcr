from classes import Data_table
import numpy as np
import cv2 as cv
from PIL import Image
import utils
from classes import Table
import xlsxwriter
import time
import os

# получение данных таблицы
def get_data_table(table, couters):
    return Data_table(table, couters).data_table

def main():
    start = time.time()

    # image_path = input("Введите путь к файлу для скаинрования:")
    image_path =  r'data.png'

    ext_img = Image.open(image_path)
    utils.mkdir("data/")
    ext_img.save("data/target.png", "PNG")
    image = cv.imread("data/target.png")

    tables = utils.search_for_tables(image)

    tabular_data = []

    for table in tables:
        contours = utils.get_contours(utils.remove_text(table))
        tabular_data = Data_table(table, contours).data_table

    utils.excel_export(tabular_data)
    print(tabular_data)
    end = time.time() - start
    print(end)
    os.system("pause")
        




if __name__ == "__main__":
    main()


