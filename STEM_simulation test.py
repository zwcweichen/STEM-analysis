import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from PyQt6.QtWidgets import *
from PyQt6.QtGui import QIcon
from ase.build import mx2
from abtem import Potential, SMatrix, AnnularDetector, GridScan
import abtem
from ase.io import read, write
from ase.build import mx2


class MyWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        self.tab1 = QWidget()
        self.tabs.addTab(self.tab1, "Tab 1")

        self.tab1_layout = QHBoxLayout()
        self.tab1.setLayout(self.tab1_layout)

        self.left_layout = QVBoxLayout()
        self.tab1_layout.addLayout(self.left_layout)

        # 新增：结构选择下拉菜单
        self.structure_combo = QComboBox()
        self.structure_combo.addItems(
            ["2H Monolayer","2H-1T Stack", "2H Bilayer", "3R Bilayer", "Twisted Bilayer"]
        )
        self.left_layout.addWidget(self.structure_combo)

        self.stemsim_buttton = QPushButton("STEM image simulation")
        self.stemsim_buttton.clicked.connect(self.STEM_simulation)
        self.left_layout.addWidget(self.stemsim_buttton)

        self.stemsim_figure, self.stemsim_ax = plt.subplots()
        self.stemsim_canvas = FigureCanvas(self.stemsim_figure)
        self.left_layout.addWidget(self.stemsim_canvas)

        self.location_label = QLabel()
        self.left_layout.addWidget(self.location_label)

        self.Save_stemsim_button = QPushButton("Save STEM image")
        self.Save_stemsim_button.clicked.connect(self.Save_STEMimage)
        self.left_layout.addWidget(self.Save_stemsim_button)

        self.right_layout = QVBoxLayout()
        self.tab1_layout.addLayout(self.right_layout)

        self.stemsim_profile_figure, self.stemsim_profile_ax = plt.subplots()
        self.stemsim_profile_canvas = FigureCanvas(self.stemsim_profile_figure)
        self.right_layout.addWidget(self.stemsim_profile_canvas)

        self.stemsim_saveprofile_button = QPushButton("Save line profile")
        self.stemsim_saveprofile_button.clicked.connect(self.Save_Lineprofile)
        self.right_layout.addWidget(self.stemsim_saveprofile_button)

        self.x0, self.y0 = None, None
        self.x1, self.y1 = None, None
        self.line = None
        self.save_lineprofilex, self.save_lineprofiley = None, None

        self.filtered_measurement = None

        self.stemsim_canvas.mpl_connect("button_press_event", self.on_press)
        self.stemsim_canvas.mpl_connect("motion_notify_event", self.on_motion)
        self.stemsim_canvas.mpl_connect("button_release_event", self.on_release)

        self.setGeometry(50, 50, 900, 600)
        self.setWindowTitle("STEM simulation GUI")
        icon = QIcon("PL Mapping_PyQt6 python/icon_mapping.png")
        self.setWindowIcon(icon)
        self.show()

    def Save_STEMimage(self):
        file_path, _ = QFileDialog.getSaveFileName(
            None, "Save jpg File", "", "JPG Files (*.jpg);;All Files (*)"
        )
        if file_path:
            if not file_path.endswith(".jpg"):
                file_path += ".jpg"
            self.stemsim_figure.savefig(file_path)
        return

    def Save_Lineprofile(self):
        df_lineprofile = pd.DataFrame(
            {
                "x_lineprofile": self.save_lineprofilex,
                "y_lineprofile": self.save_lineprofiley,
            }
        )
        file_path, _ = QFileDialog.getSaveFileName(
            None, "Save Excel File", "", "Excel Files (*.xlsx);;All Files (*)"
        )
        if file_path:
            if not file_path.endswith(".xlsx"):
                file_path += ".xlsx"
            with pd.ExcelWriter(file_path) as writer:
                df_lineprofile.to_excel(
                    writer, sheet_name="Lineprofile Data", index=False
                )
        return

    def STEM_simulation(self):
        selected_structure = self.structure_combo.currentText()

        if selected_structure == "2H-1T Stack":
            atoms = self.create_2H_1T_stack()
        elif selected_structure == "2H Bilayer":
            atoms = self.create_2H_bilayer()
        elif selected_structure == "3R Bilayer":
            atoms = self.create_3R_bilayer()
        elif selected_structure == "Twisted Bilayer":
            atoms = self.create_twisted_bilayer()
        elif selected_structure == "2H Monolayer":
            atoms = self.create_2H_ML()


        # 调用 abtem 模拟
        device = "cpu"  # or 'cpu'

        # 定义电势
        potential = Potential(
            atoms,
            gpts=512,
            projection="infinite",
            slice_thickness=1,
            parametrization="kirkland",
            device=device,
        )
        potential = potential.build()

        # 定义探针
        probe = abtem.Probe(
            energy=80e3, semiangle_cutoff=25, Cs=10e4, defocus="scherzer"
        )

        # 定义探测器
        detector = abtem.FlexibleAnnularDetector()

        # 定义扫描区域
        repetitions = (1, 1, 1)
        end = (
            4 * potential.extent[0] / repetitions[0],
            4 * potential.extent[0] / repetitions[0],
        )
        grid_scan = abtem.GridScan(
            start=[0, 0],
            end=end,
            sampling=probe.aperture.nyquist_sampling,
            potential=potential,
        )

        # 运行模拟
        flexible_measurement = probe.scan(
            scan=grid_scan, detectors=detector, potential=potential
        )

        flexible_measurement.compute()

        bf_measurement = flexible_measurement.integrate_radial(
            0, probe.semiangle_cutoff
        )
        maadf_measurement = flexible_measurement.integrate_radial(50, 150)
        haadf_measurement = flexible_measurement.integrate_radial(90, 200)

        measurements = abtem.stack(
            [bf_measurement, maadf_measurement, haadf_measurement],
            ("BF", "MAADF", "HAADF"),
        )

        interpolated_measurement = measurements.interpolate(0.05)
        self.filtered_measurement = interpolated_measurement.gaussian_filter(0.3)

        self.filtered_measurement[-1].show(
            ax=self.stemsim_ax,
            explode=True,
            figsize=(14, 5),
            cbar=True,
            cmap="gray",  # 设置为灰度显示
        )
        self.stemsim_ax.axes.xaxis.set_visible(False)
        self.stemsim_ax.axes.yaxis.set_visible(False)
        return
    
    def create_2H_ML(self):
        repetitions = (3, 3, 1)
        atoms_2H = mx2(
            formula="MoS2",
            kind="2H",
            a=3.18,
            thickness=3.19,
            size=(1, 1, 1),
            vacuum=None,
        )
        atoms_2H *= repetitions
        atoms_2H.center(vacuum=2, axis=2)
        return atoms_2H



    def create_2H_1T_stack(self):
        repetitions = (3, 3, 1)
        atoms_2H = mx2(
            formula="MoS2",
            kind="2H",
            a=3.18,
            thickness=3.19,
            size=(1, 1, 1),
            vacuum=None,
        )
        atoms_2H *= repetitions
        atoms_2H.center(vacuum=2, axis=2)
        

        atoms_1T = mx2(
            formula="MoS2",
            kind="1T",
            a=3.17,
            thickness=3.19,
            size=(1, 1, 1),
            vacuum=None,
        )
        atoms_1T *= repetitions
        atoms_1T.center(vacuum=2, axis=2)

        atoms_1T.positions[:, 2] += (
            atoms_2H.positions[:, 2].max() - atoms_1T.positions[:, 2].min() + 3.19
        )

        stacked_atoms = atoms_2H + atoms_1T

        stacked_atoms.set_cell(
            [atoms_2H.cell[0], atoms_2H.cell[1], atoms_2H.cell[2] + atoms_1T.cell[2]]
        )
        return stacked_atoms

    def create_2H_bilayer(self):
        filename = "C:/研究所/實驗室/光學/資料/20240620_STEM/ZWC/參考資料/新增資料夾/MoS2_BL_2H.cif"
        return read(filename)

    def create_3R_bilayer(self):
        filename = "C:/git/STEM analysis/atomic model/MoS2_2L_3R_VESTA edit.cif"
        return read(filename)

    def create_twisted_bilayer(self):
        size = (1, 1, 1)
        layer1 = mx2(formula="MoS2", kind="2H", a=3.18, thickness=3.19, size=size)
        layer2 = layer1.copy()

        rotation_angle = 25  # 度
        layer2.rotate(rotation_angle, "z", center="COM")

        layer2.translate([0, 0, 6.5])  # 調整層間距
        bilayer = layer1 + layer2
        bilayer.center(vacuum=5, axis=2)
        output_file = "C:/研究所/實驗室/光學/資料/20240620_STEM/ZWC/參考資料/新增資料夾/twisted_BL.cif"
        write(output_file, bilayer, format="cif")
        return bilayer

    def on_press(self, event):
        if (
            event.button == 1 and event.inaxes == self.stemsim_ax
        ):  # 左键按下且在axes范围内
            self.x0, self.y0 = event.xdata, event.ydata
            if self.line is not None:
                self.line.remove()
                self.line = None
            (self.line,) = self.stemsim_ax.plot(
                [self.x0, self.x0], [self.y0, self.y0], "r-"
            )
            self.stemsim_canvas.draw()
            self.stemsim_profile_ax.cla()

    def on_motion(self, event):
        if (
            event.button == 1
            and self.line is not None
            and event.inaxes == self.stemsim_ax
        ):
            self.x1, self.y1 = event.xdata, event.ydata
            self.line.set_data([self.x0, self.x1], [self.y0, self.y1])
            self.location_label.setText(
                f"(x0: {self.x0:.3f}, y0: {self.y0:.3f}), (x1: {self.x1:.3f}, y1: {self.y1:.3f})"
            )

            self.stemsim_profile_ax.cla()

            line_profile = self.filtered_measurement.interpolate_line(
                start=(self.x0, self.y0),
                end=(self.x1, self.y1),
            )

            line_profile[-1].show(ax=self.stemsim_profile_ax)
            self.stemsim_canvas.draw()
            self.stemsim_profile_canvas.draw()

    def on_release(self, event):
        if (
            event.button == 1
            and self.line is not None
            and event.inaxes == self.stemsim_ax
        ):
            self.x1, self.y1 = event.xdata, event.ydata
            self.line.set_data([self.x0, self.x1], [self.y0, self.y1])
            self.location_label.setText(
                f"(x0: {self.x0:.3f}, y0: {self.y0:.3f}), (x1: {self.x1:.3f}, y1: {self.y1:.3f})"
            )
            self.stemsim_profile_ax.cla()
            line_profile = self.filtered_measurement.interpolate_line(
                start=(self.x0, self.y0),
                end=(self.x1, self.y1),
            )
            line_profile[-1].show(ax=self.stemsim_profile_ax)
            self.save_lineprofiley = line_profile.array[-1]
            self.save_lineprofilex = np.linspace(
                0,
                np.sqrt((abs(self.x0 - self.x1)) ** 2 + (abs(self.y0 - self.y1)) ** 2),
                np.shape(self.save_lineprofiley)[0],
            )

            self.stemsim_profile_canvas.draw()
            self.stemsim_canvas.draw()
            self.x0, self.y0 = None, None


def main():
    app = QApplication(sys.argv)
    window = MyWindow()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
