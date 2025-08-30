import tkinter as tk
import requests

class URLCheckerApp(tk.Tk):
    # initialize GUI window with text, search box, and search button
    def __init__(self):
        super().__init__()
        # main window formatting
        self.title("URL Live Checker")
        self.geometry("400x200")

        # instruction label and formatting
        self.label = tk.Label(self, text="Enter a URL to check:")
        self.label.pack(pady=10)

        # search box and formatting
        self.url_entry = tk.Entry(self, width=50)
        self.url_entry.pack(pady=5)
        # allow enter key to comlete search (binds return key to check_url method)
        self.url_entry.bind('<Return>', lambda event: self.check_url())

        # button to check URL and formatting
        self.check_button = tk.Button(self, text="Check URL", command=self.check_url)
        self.check_button.pack(pady=10)

        # label to display results of checking URL
        self.result_label = tk.Label(self, text="", font=("Helvetica", 12))
        self.result_label.pack(pady=10)

    # method to verify status of URL
    def check_url(self):
        # ensure no extra whitespace in URL
        url = self.url_entry.get().strip()

        # ensure URL includes http
        if not url.startswith(('http://', 'https://')):
            url = 'http://' + url

        try:
            # send head request to URL
            response = requests.head(url, allow_redirects=True, timeout=5)
            status = response.status_code
            # if response is recieved, determine whether status code was successful
            if 200 <= status < 400:
                self.result_label.config(
                    text=f"{url} is online (HTTP {status})", fg="green")
            else:
                self.result_label.config(
                    text=f"{url} responded (HTTP {status})", fg="orange")
        # otherwise, print error message
        except requests.RequestException as e:
            self.result_label.config(
                text=f"{url} is unreachable: {e.__class__.__name__}", fg="red")

# run the app
if __name__ == "__main__":
    app = URLCheckerApp()
    app.mainloop()
