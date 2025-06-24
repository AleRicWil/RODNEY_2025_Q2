import tkinter as tk
from tkinter import ttk
import serial.tools.list_ports
from collect_data import run_collection
from calibration import run_calibration
from multiprocessing import Process, Queue
import time
from datetime import datetime

class HardwareControlUI:
    """Main UI class for controlling RODNEY hardware and collecting strain data.

    Attributes:
        root (tk.Tk): The main Tkinter window.
        container (ttk.Frame): Frame to hold different pages.
        pages (dict): Dictionary mapping page names to their frames.
        status_queue (Queue): Queue for receiving status messages from collection process.
        collection_process (Process): Current data collection process, if running.
    """

    def __init__(self, root):
        """Initialize the UI.

        Args:
            root (tk.Tk): The Tkinter root window.
        """
        self.root = root
        self.root.title("RODNEY Hardware Control")
        self.root.geometry("400x400")

        self.container = ttk.Frame(self.root)
        self.container.pack(fill="both", expand=True)

        self.pages = {}
        self.create_home_page()
        self.create_collect_data_page()
        self.create_calibrate_page()

        self.show_page("Home")

        self.status_queue = Queue()
        self.collection_process = None

    def create_home_page(self):
        """Create the home page with navigation buttons."""
        page = ttk.Frame(self.container)
        self.pages["Home"] = page

        ttk.Label(page, text="Welcome to RODNEY Control", font=("Arial", 16)).pack(pady=20)
        ttk.Button(page, text="Collect Data", command=lambda: self.show_page("Collect Data")).pack(pady=10)
        ttk.Button(page, text="Calibrate", command=lambda: self.show_page("Calibrate")).pack(pady=10)
        ttk.Button(page, text="Process Data", command=lambda: self.show_page("Process Data")).pack(pady=10)

    def create_collect_data_page(self):
        """Create the data collection page with controls."""
        page = ttk.Frame(self.container)
        self.pages["Collect Data"] = page

        ttk.Label(page, text="Arduino Connection", font=("Arial", 14)).pack(pady=10)
        self.connect_button = ttk.Button(page, text="Connect to Arduino", command=self.connect_arduino)
        self.connect_button.pack(pady=5)

        self.available_ports = [port.device for port in serial.tools.list_ports.comports()]
        self.port_var = tk.StringVar()
        self.port_combobox = ttk.Combobox(page, textvariable=self.port_var, values=self.available_ports, state="readonly")
        self.port_combobox.pack(pady=5)
        if self.available_ports:
            self.port_var.set(self.available_ports[0])

        ttk.Label(page, text="Date (MM_DD)", font=("Arial", 10)).pack(pady=5)
        self.month_date_var = tk.StringVar(value=datetime.now().strftime("%m_%d"))
        self.month_date_entry = ttk.Entry(page, textvariable=self.month_date_var, width=10)
        self.month_date_entry.pack(pady=5)

        ttk.Label(page, text="Test Number", font=("Arial", 10)).pack(pady=5)
        self.test_num_var = tk.StringVar(value="1")
        self.test_num_entry = ttk.Entry(page, textvariable=self.test_num_var, width=10)
        self.test_num_entry.pack(pady=5)

        self.start_button = ttk.Button(page, text="Start Data Collection", command=self.start_collection, state="disabled")
        self.start_button.pack(pady=5)

        self.status_var = tk.StringVar(value="Status: Idle")
        ttk.Label(page, textvariable=self.status_var).pack(pady=10)

    def create_calibrate_page(self):
        """Create the calibrate page with controls."""
        page = ttk.Frame(self.container)
        self.pages["Calibrate"] = page

        ttk.Label(page, text="Arduino Connection", font=("Arial", 14)).pack(pady=10)
        self.cal_connect_button = ttk.Button(page, text="Connect to Arduino", command=self.connect_arduino_cal)
        self.cal_connect_button.pack(pady=5)

        self.cal_available_ports = [port.device for port in serial.tools.list_ports.comports()]
        self.cal_port_var = tk.StringVar()
        self.cal_port_combobox = ttk.Combobox(page, textvariable=self.cal_port_var, values=self.cal_available_ports, state="readonly")
        self.cal_port_combobox.pack(pady=5)
        if self.cal_available_ports:
            self.cal_port_var.set(self.cal_available_ports[0])

        ttk.Label(page, text="Date (MM_DD)", font=("Arial", 10)).pack(pady=5)
        self.cal_month_date_var = tk.StringVar(value=datetime.now().strftime("%m_%d"))
        self.cal_month_date_entry = ttk.Entry(page, textvariable=self.cal_month_date_var, width=10)
        self.cal_month_date_entry.pack(pady=5)

        ttk.Label(page, text="Weights (grams, space-separated)", font=("Arial", 10)).pack(pady=5)
        self.weights_var = tk.StringVar()
        self.weights_entry = ttk.Entry(page, textvariable=self.weights_var, width=20)
        self.weights_entry.pack(pady=5)

        ttk.Label(page, text="Positions (cm, space-separated)", font=("Arial", 10)).pack(pady=5)
        self.positions_var = tk.StringVar()
        self.positions_entry = ttk.Entry(page, textvariable=self.positions_var, width=20)
        self.positions_entry.pack(pady=5)

        self.calibrate_button = ttk.Button(page, text="Start Calibration", command=self.start_calibration, state="disabled")
        self.calibrate_button.pack(pady=5)

        self.cal_status_var = tk.StringVar(value="Status: Idle")
        ttk.Label(page, textvariable=self.cal_status_var).pack(pady=10)

    def show_page(self, page_name):
        """Display the specified page.

        Args:
            page_name (str): Name of the page to show.
        """
        for page in self.pages.values():
            page.pack_forget()
        if page_name in self.pages:
            self.pages[page_name].pack(fill="both", expand=True)
        if page_name == "Collect Data":
            self.available_ports = [port.device for port in serial.tools.list_ports.comports()]
            self.port_combobox["values"] = self.available_ports
            if self.available_ports:
                self.port_var.set(self.available_ports[0])
                self.connect_button["state"] = "normal"
            else:
                self.status_var.set("Status: No COM ports found")
                self.connect_button["state"] = "disabled"
            self.start_button["state"] = "disabled"
            self.month_date_var.set(datetime.now().strftime("%m_%d"))
            self.test_num_var.set("1")
        elif page_name == "Calibrate":
            self.cal_available_ports = [port.device for port in serial.tools.list_ports.comports()]
            self.cal_port_combobox["values"] = self.cal_available_ports
            if self.cal_available_ports:
                self.cal_port_var.set(self.cal_available_ports[0])
                self.cal_connect_button["state"] = "normal"
            else:
                self.cal_status_var.set("Status: No COM ports found")
                self.cal_connect_button["state"] = "disabled"
            self.calibrate_button["state"] = "disabled"
            self.cal_month_date_var.set(datetime.now().strftime("%m_%d"))

    def connect_arduino(self):
        """Enable data collection after selecting a port."""
        self.connect_button["state"] = "disabled"
        self.root.update()
        time.sleep(1)
        selected_port = self.port_var.get()
        if not selected_port:
            self.status_var.set("Status: No COM port selected")
            self.connect_button["state"] = "normal"
            return
        self.start_button["state"] = "normal"
        self.status_var.set(f"Status: Ready to collect data from {selected_port}")

    def connect_arduino_cal(self):
        """Enable calibration after selecting a port."""
        self.cal_connect_button["state"] = "disabled"
        self.root.update()
        time.sleep(1)
        selected_port = self.cal_port_var.get()
        if not selected_port:
            self.cal_status_var.set("Status: No COM port selected")
            self.cal_connect_button["state"] = "normal"
            return
        self.calibrate_button["state"] = "normal"
        self.cal_status_var.set(f"Status: Ready to calibrate from {selected_port}")

    def start_collection(self):
        """Start data collection in a separate process."""
        self.start_button["state"] = "disabled"
        self.root.update()
        time.sleep(1)
        selected_port = self.port_var.get()
        month_date = self.month_date_var.get()
        test_num = self.test_num_var.get()

        if not month_date or not test_num:
            self.status_var.set("Status: Please enter date and test number")
            self.start_button["state"] = "normal"
            return
        if not month_date.replace("_", "").isdigit() or len(month_date) != 5 or month_date[2] != "_":
            self.status_var.set("Status: Invalid date format (use MM_DD)")
            self.start_button["state"] = "normal"
            return
        if not test_num.isdigit():
            self.status_var.set("Status: Test number must be a number")
            self.start_button["state"] = "normal"
            return

        config = {
            "date": month_date,
            "test_num": int(test_num),
            "configuration": "Config 1",
            "pvc_stiffness": "Medium",
            "height": 80.645,
            "yaw": 5,
            "pitch": 0,
            "roll": 0,
            "rate_of_travel": 25,
            "angle_of_travel": 0,
            "offset_distance": 25,
        }
        self.collection_process = Process(target=run_collection, args=(selected_port, config, self.status_queue))
        self.collection_process.start()
        self.check_status_queue()

    def start_calibration(self):
        """Start calibration in a separate process."""
        self.calibrate_button["state"] = "disabled"
        self.root.update()
        time.sleep(1)
        selected_port = self.cal_port_var.get()
        month_date = self.cal_month_date_var.get()
        weights = self.weights_var.get()
        positions = self.positions_var.get()

        if not month_date or not weights or not positions:
            self.cal_status_var.set("Status: Please enter all fields")
            self.cal_connect_button["state"] = "normal"
            self.calibrate_button["state"] = "normal"
            return
        if not month_date.replace("_", "").isdigit() or len(month_date) != 5 or month_date[2] != "_":
            self.cal_status_var.set("Status: Invalid date format (use MM_DD)")
            self.cal_connect_button["state"] = "normal"
            self.calibrate_button["state"] = "normal"
            return

        try:
            weights_list = [float(w) for w in weights.split()]
            positions_list = [float(p) for p in positions.split()]
            if not weights_list or not positions_list:
                raise ValueError
        except ValueError:
            self.cal_status_var.set("Status: Weights and positions must be numbers")
            self.cal_connect_button["state"] = "normal"
            self.calibrate_button["state"] = "normal"
            return

        config = {
            "date": month_date,
            "weights": weights_list,
            "positions": positions_list,
            "configuration": "Config 1",
            "pvc_stiffness": "Medium",
            "height": 80.645,
            "yaw": 5,
            "pitch": 0,
            "roll": 0,
            "rate_of_travel": 25,
            "angle_of_travel": 0,
            "offset_distance": 25,
        }
        self.collection_process = Process(target=run_calibration, args=(selected_port, config, self.status_queue))
        self.collection_process.start()
        self.check_status_queue()

    def reset_collect_data_page(self):
        """Reset the Collect Data page to initial state for a new test."""
        self.connect_button["state"] = "normal" if self.available_ports else "disabled"
        self.start_button["state"] = "normal" if self.port_var.get() else "disabled"
        self.month_date_var.set(datetime.now().strftime("%m_%d"))
        self.test_num_var.set(str(int(self.test_num_var.get()) + 1))
        self.status_var.set("Status: Ready for new test")
        self.collection_process = None

    def reset_calibrate_page(self):
        """Reset the Calibrate page to initial state for a new calibration."""
        self.cal_connect_button["state"] = "normal" if self.cal_available_ports else "disabled"
        self.calibrate_button["state"] = "normal" if self.cal_port_var.get() else "disabled"
        self.cal_month_date_var.set(datetime.now().strftime("%m_%d"))
        self.weights_var.set("")
        self.positions_var.set("")
        self.cal_status_var.set("Status: Ready for new calibration")
        self.collection_process = None

    def check_status_queue(self):
        """Periodically check the status queue and update the UI."""
        while not self.status_queue.empty():
            message = self.status_queue.get()
            if self.pages["Collect Data"].winfo_ismapped():
                self.status_var.set(f"Status: {message}")
                if message == "Data collection ended":
                    self.reset_collect_data_page()
            elif self.pages["Calibrate"].winfo_ismapped():
                self.cal_status_var.set(f"Status: {message}")
                if message == "Calibration ended":
                    self.reset_calibrate_page()
        self.root.after(100, self.check_status_queue)

if __name__ == "__main__":
    root = tk.Tk()
    app = HardwareControlUI(root)
    root.mainloop()