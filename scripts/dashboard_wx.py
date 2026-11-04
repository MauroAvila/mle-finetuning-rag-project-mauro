import json
import wx
from pathlib import Path

class HybridMetricsDashboard(wx.Frame):
    def __init__(self, parent, title):
        super(HybridMetricsDashboard, self).__init__(parent, title=title, size=(950, 750))
        
        # Load evaluation metrics data using the exact provided path
        self.metrics_data = self.load_metrics()
        
        # Initialize user interface elements with accessibility in mind
        self.init_ui()
        self.update_display()
        
        self.Center()

    def load_metrics(self):
        # Explicit path provided by user
        explicit_path = Path(r"C:\Users\Mauro\resources\mod5\mle-finetuning-rag-project-mauro\data\hito3_hybrid_output\metrics\hito3_hybrid_metrics.json")
        
        # Comprehensive list including the explicit path and relative fallbacks
        possible_paths = [
            explicit_path,
            Path("data/hito3_hybrid_output/metrics/hito3_hybrid_metrics.json"),
            Path("metrics/hito3_hybrid_metrics.json"),
            Path("hito3_hybrid_output/metrics/hito3_hybrid_metrics.json"),
            Path("../hito3_hybrid_output/metrics/hito3_hybrid_metrics.json")
        ]
        
        for path in possible_paths:
            resolved_path = path.resolve() if not path.is_absolute() else path
            if resolved_path.exists():
                print(f"[INFO] Metrics file successfully loaded from: {resolved_path}")
                try:
                    with open(resolved_path, "r", encoding="utf-8") as f:
                        return json.load(f)
                except Exception as e:
                    print(f"[ERROR] Failed to parse JSON from {resolved_path}: {e}")
                    
        print(f"[WARNING] 'hito3_hybrid_metrics.json' was NOT found. Checked path: {explicit_path}")
        return {
            "total_cases": 0,
            "successful_cases": 0,
            "accuracy": 0.0,
            "details": []
        }

    def init_ui(self):
        panel = wx.Panel(self)
        main_sizer = wx.BoxSizer(wx.VERTICAL)

        # -------------------------------------------------------------
        # 1. Control Parameters and Filters Box
        # -------------------------------------------------------------
        self.control_box = wx.StaticBox(panel, label="Control Parameters and Filters")
        control_sizer = wx.StaticBoxSizer(self.control_box, wx.VERTICAL)
        
        grid_sizer = wx.FlexGridSizer(rows=2, cols=3, hgap=10, vgap=10)

        # Controls must be children of the StaticBox for optimal screen reader accessibility
        self.filter_label = wx.StaticText(self.control_box, label="Filter by result:")
        grid_sizer.Add(self.filter_label, 0, wx.ALIGN_CENTER_VERTICAL | wx.LEFT, 5)

        filter_choices = ["All", "Only Successful", "Only Failed"]
        self.filter_combo = wx.ComboBox(self.control_box, value="All", choices=filter_choices, style=wx.CB_READONLY)
        self.filter_combo.SetToolTip("Filter evaluation items by their execution status")
        grid_sizer.Add(self.filter_combo, 0, wx.EXPAND | wx.ALIGN_CENTER_VERTICAL, 5)
        
        grid_sizer.Add(wx.StaticText(self.control_box, label=""), 0)

        self.search_label = wx.StaticText(self.control_box, label="Search keyword:")
        grid_sizer.Add(self.search_label, 0, wx.ALIGN_CENTER_VERTICAL | wx.LEFT, 5)

        self.search_entry = wx.TextCtrl(self.control_box, value="", style=wx.TE_PROCESS_ENTER)
        self.search_entry.SetToolTip("Type a keyword to filter IDs, questions, answers, or RAG context")
        grid_sizer.Add(self.search_entry, 0, wx.EXPAND | wx.ALIGN_CENTER_VERTICAL, 5)

        self.refresh_btn = wx.Button(self.control_box, label="Refresh View")
        self.refresh_btn.SetToolTip("Click or press Enter to apply filters and update view")
        grid_sizer.Add(self.refresh_btn, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 5)

        control_sizer.Add(grid_sizer, 0, wx.EXPAND | wx.ALL, 10)
        main_sizer.Add(control_sizer, 0, wx.EXPAND | wx.ALL, 10)

        # -------------------------------------------------------------
        # 2. Consolidated Summary Box
        # -------------------------------------------------------------
        self.stats_box = wx.StaticBox(panel, label="Consolidated Summary")
        stats_sizer = wx.StaticBoxSizer(self.stats_box, wx.VERTICAL)

        self.stats_text = wx.StaticText(self.stats_box, label="Initializing metrics summary...")
        stats_sizer.Add(self.stats_text, 0, wx.ALL, 10)
        
        main_sizer.Add(stats_sizer, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 10)

        # -------------------------------------------------------------
        # 3. Case Details and Inferences Box (Read-Only Text Control)
        # -------------------------------------------------------------
        self.details_box = wx.StaticBox(panel, label="Case Details and Inferences (Read-Only)")
        details_sizer = wx.StaticBoxSizer(self.details_box, wx.VERTICAL)

        self.text_output = wx.TextCtrl(
            self.details_box, 
            style=wx.TE_MULTILINE | wx.TE_READONLY | wx.TE_RICH2 | wx.HSCROLL
        )
        self.text_output.SetFont(wx.Font(10, wx.FONTFAMILY_TELETYPE, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL))
        
        details_sizer.Add(self.text_output, 1, wx.EXPAND | wx.ALL, 5)
        main_sizer.Add(details_sizer, 1, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 10)

        panel.SetSizer(main_sizer)

        # Bind Events
        self.filter_combo.Bind(wx.EVT_COMBOBOX, self.on_filter_changed)
        self.search_entry.Bind(wx.EVT_TEXT, self.on_search_typed)
        self.refresh_btn.Bind(wx.EVT_BUTTON, self.on_refresh_clicked)

    def on_filter_changed(self, event):
        self.update_display()

    def on_search_typed(self, event):
        self.update_display()

    def on_refresh_clicked(self, event):
        self.update_display()

    def update_display(self):
        self.text_output.Clear()

        total = self.metrics_data.get("total_cases", 0)
        success = self.metrics_data.get("successful_cases", 0)
        accuracy = self.metrics_data.get("accuracy", 0.0)
        details = self.metrics_data.get("details", [])

        summary_str = f"Total Evaluated: {total} | Successful: {success} | Global Accuracy: {accuracy * 100:.2f}%"
        self.stats_text.SetLabel(summary_str)

        filter_option = self.filter_combo.GetValue()
        search_query = self.search_entry.GetValue().strip().lower()

        filtered_count = 0
        output_buffer = []

        for item in details:
            passed = item.get("passed", False)
            
            if filter_option == "Only Successful" and not passed:
                continue
            if filter_option == "Only Failed" and passed:
                continue

            case_id = str(item.get("id", "")).lower()
            question = str(item.get("question", "")).lower()
            response = str(item.get("response", "")).lower()
            rag_context = str(item.get("retrieved_context", "")).lower()

            if search_query:
                match_found = (
                    search_query in case_id or
                    search_query in question or
                    search_query in response or
                    search_query in rag_context
                )
                if not match_found:
                    continue

            filtered_count += 1
            status_str = "SUCCESS" if passed else "FAILED"
            
            case_block = (
                f"--------------------------------------------------\n"
                f"Case ID: {item.get('id')} [{status_str}]\n"
                f"Question: {item.get('question')}\n"
                f"Generated Response: {item.get('response')}\n"
                f"Retrieved RAG Context: {item.get('retrieved_context')}\n\n"
            )
            output_buffer.append(case_block)

        if filtered_count == 0:
            output_buffer.append("No cases found matching the selected filters.\n")

        self.text_output.SetValue("".join(output_buffer))

if __name__ == "__main__":
    app = wx.App(False)
    frame = HybridMetricsDashboard(None, title="Hito 3 Analytical Dashboard - wxPython Accessible GUI")
    frame.Show()
    app.MainLoop()