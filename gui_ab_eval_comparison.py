from nicegui import app, binding, ui, events, elements

class ABEvalGUI:
    """
    A GUI for side-by-side comparison of evaluations run under two different conditions.
    """

    def __init__(self):
        """Initialize the GUI."""
        # set up state
        # set up GUI
        self.dark_setting = ui.dark_mode(value=True)
        self.setup_ui()

    def setup_ui(self):
        """Build the GUI itself."""
        # define navigation tabs
        with ui.tabs().classes('w-full') as tabs:
            tab_a = ui.tab('Condition A')
            self.tab_b = ui.tab('Condition B')
            self.tab_comparison = ui.tab('Comparison')
            self.tab_settings = ui.tab("Settings")
        # define contents of each tab
        with ui.tab_panels(tabs, value=tab_a).classes('w-7/8'):
            # ---------- CONDITION A TAB ----------------
            self.panel_a = ui.tab_panel(tab_a)
            with self.panel_a:
                # file uploader to select the evaluation results we want to look at
                eval_result_uploader = ui.upload(
                    on_upload=self.handle_upload,
                    max_file_size=10e6,
                    multiple=False,
                    # max_files=1,
                    auto_upload=True,
                    label="Upload condition A results:"
                )
                eval_result_uploader.props('accept=.csv')
            
            # --------------- SETTINGS TAB -------------------
            
            with ui.tab_panel(self.tab_settings):
                # toggle to control light/dark theme
                ui.switch('Dark mode').bind_value(self.dark_setting)
    
    async def handle_upload(e: events.UploadEventArguments):
        """
        Uploads evaluation data file and adds the data to the table.
        """
        topic_display_text = await e.file.text()
        # convert the CSV text into a data frame
        eval_results_data = pd.read_csv(io.StringIO(topic_display_text))

        result_rows = []
        for row in eval_results_data.itertuples():
            row_data = {
                "pubmed_id": row.pubmed_id,
                "date": row.date,
                "title": row.title,
                "source": row.source,
                "is_relevant": row.is_relevant,
                "gold_standard": row.gold_standard,
                "abstract": row.abstract,
                "query": row.query,
                "evaluation": row.evaluation,
            }
            result_rows.append(row_data)
        table_results_data.rows = result_rows

        # calculate whole-dataset statistics
        y_true = eval_results_data['gold_standard'].to_numpy(dtype=np.bool)
        y_pred = eval_results_data['is_relevant'].to_numpy(dtype=np.bool)
        cm = confusion_matrix(
            y_true=y_true, y_pred=y_pred,
        )
        table_conf_mat.rows = [
            {'row_label': 'Irrelevant', 'negative': cm[0, 0], 'positive': cm[0, 1]},
            {'row_label': 'Relevant', 'negative': cm[1, 0], 'positive': cm[1, 1]},
        ]

        # calculate accuracy
        pred_accuracy = np.sum(y_true == y_pred)/len(y_true)
        label_accuracy.text = f"Accuracy: {pred_accuracy:.1%}"
        # calculate PPV
        true_positives = np.sum((y_true == 1) & (y_pred == 1))
        false_positives = np.sum((y_true == 0) & (y_pred == 1))
        if (true_positives + false_positives) > 0:
            ppv = true_positives/(true_positives + false_positives)
        else:
            ppv = 0.0
        label_ppv.text = f"PPV: {ppv:.1%}"
        # calculate NPV
        true_negatives = np.sum((y_true == 0) & (y_pred == 0))
        false_negatives = np.sum((y_true == 1) & (y_pred == 0))
        if (true_negatives + false_negatives) > 0:
            npv = true_negatives/(true_negatives + false_negatives)
        else:
            npv = 0.0
        label_npv.text = f"NPV: {npv:.1%}"

gui = ABEvalGUI()
ui.run(host='127.0.0.1', port=9092, title="New Lit A/B Eval")