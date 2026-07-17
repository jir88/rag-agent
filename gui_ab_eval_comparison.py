from nicegui import ui, events, elements

import sys

from monitor import Article,LitMonitorState

class ABEvalGUI:
    """
    A GUI for side-by-side comparison of evaluations run under two different conditions.
    """

    # ===== Reference Results =============
    ref_agent_results: LitMonitorState
    ref_current_article: Article
    """Currently selected article, if any."""

    # GUI elements
    ref_ta_system_prompt: elements.textarea.Textarea
    """Text area showing the system prompt used by the agent."""
    ref_ta_relevance_prompt: elements.textarea.Textarea
    """Text area showing the format used to present articles for evaluation."""

    ref_table_results_data: elements.table.Table
    """Table showing articles in the current result."""
    ref_label_title: elements.label.Label
    """Label to hold selected article title."""
    ref_label_abstract: elements.label.Label
    """Label to hold selected article abstract."""
    ref_label_query: elements.label.Label
    """Label to hold the criteria for whether an article is relevant."""
    ref_cb_article_relevant: elements.checkbox.Checkbox
    """Check box showing/controlling whether article is judged as relevant."""
    ref_ta_article_eval: elements.textarea.Textarea
    """Text area showing the article relevance evaluation."""

    def __init__(self):
        """Initialize the GUI."""
        # set up state
        # set up GUI
        self.dark_setting = ui.dark_mode(value=True)
        self.setup_ui()

    def setup_ui(self):
        """Build the GUI itself."""
        # define navigation tabs
        with ui.header().classes('bg-dark'):
            with ui.tabs().classes('w-full') as tabs:
                tab_ref = ui.tab('Reference')
                self.tab_memory = ui.tab('Condition A')
                self.tab_archive = ui.tab('Condition B')
                self.tab_settings = ui.tab("Comparison")
        # define contents of each tab
        with ui.tab_panels(tabs, value=tab_ref).classes('w-7/8'):

            # ------ MAIN TAB -------------

            ref_panel = ui.tab_panel(tab_ref)
            with ref_panel:
                with ui.row():
                    # file uploader to select the evaluation results we want to look at
                    ref_eval_result_uploader = ui.upload(
                        on_upload=self.handle_ref_upload,
                        max_file_size=10e6,
                        multiple=False,
                        max_files=1,
                        auto_upload=True,
                        label="Upload evaluation results:"
                    )
                    ref_eval_result_uploader.props('accept=.json')

                    ui.button(text="Save", icon='save', on_click=self.handle_ref_save)
                
                # common settings for all articles

                ui.label("System prompt:").classes("text-2xl")
                self.ref_ta_system_prompt = ui.textarea(
                    placeholder="Agent's system prompt.",
                    on_change=self.handle_ref_prompt_update
                ).classes("text-base w-7/8")

                ui.label("Article relevance prompt:").classes("text-2xl")
                self.ref_ta_relevance_prompt = ui.textarea(
                    placeholder="Prompt used to present articles for evaluation.",
                    on_change=self.handle_ref_prompt_update
                ).classes("text-base w-7/8")

                # table showing the articles in this evaluation run
                columns = [
                    {'name': 'title', 'label': 'Title', 'field': 'title', 'required': True, 'align': 'left'},
                    {'name': 'date', 'label': 'Published', 'field': 'date', 'sortable': True},
                    {'name': 'is_relevant', 'label': 'Relevant?', 'field':'is_relevant', 'sortable': True}
                ]
                self.ref_table_results_data = ui.table(
                    rows=[], 
                    columns=columns,
                    selection='single', 
                    row_key='pubmed_id',
                    pagination=3,
                    on_select=self.handle_ref_result_selection,
                ).classes("w-7/8")

                # information about an individual article

                self.ref_label_title = ui.label("Title").classes("text-3xl w-7/8")
                self.ref_label_abstract = ui.label().classes("text-base w-7/8")
                # the criteria for article relevance
                ui.label("Relevance criteria:").classes("text-2xl")
                self.ref_label_query = ui.label().classes("text-base w-7/8")
                # what the LLM thought about the article

                self.ref_cb_article_relevant = ui.checkbox(
                    text="Article relevant to query",
                    value=False,
                    on_change=self.handle_ref_result_update
                )
                ui.label("Why is/isn't the article relevant?").classes("text-2xl")
                self.ref_ta_article_eval = ui.textarea(
                    placeholder="Write explanation here.",
                    on_change=self.handle_ref_result_update
                ).classes("text-base w-7/8")
            
            # --------------- SETTINGS TAB -------------------
            
            with ui.tab_panel(self.tab_settings):
                # toggle to control light/dark theme
                ui.switch('Dark mode').bind_value(self.dark_setting)
    
    async def handle_ref_upload(self, e: events.UploadEventArguments):
        """
        Uploads an agent result file and loads the data into the reference results table.

        Args:
            e: The file upload event.
        """
        # Read the result file
        try:
            self.ref_agent_results = LitMonitorState.model_validate_json(await e.file.text())
        except Exception as e:
            ui.notify(
                message=f"Error reading evaluation file: {e}",
                type='warning',
                multi_line=True
            )
            sys.exit(1)
        
        # clear the upload widget
        e.sender.reset()

        # load prompts
        self.ref_ta_system_prompt.value = self.ref_agent_results.agent_system_prompt
        self.ref_ta_relevance_prompt.value = self.ref_agent_results.article_relevance_prompt
        
        # populate the table
        result_rows = []
        index = 0
        for article in self.ref_agent_results.new_articles:
            row_data = {
                "index": index,
                "pubmed_id": article.pubmed_id,
                "date": article.date,
                "title": article.title,
                "source": article.source,
                "is_relevant": article.is_relevant,
                "abstract": article.abstract,
                "query": self.ref_agent_results.topic_description,
                "evaluation": article.evaluation
            }
            result_rows.append(row_data)
            index += 1
        self.ref_table_results_data.rows = result_rows

    def handle_ref_save(self):
        """Save the monitor results to a JSON file."""
        if self.agent_results is None:
            return
        output_file_txt = self.ref_agent_results.model_dump_json(indent=2)
        # show save dialog
        ui.download.content(
            content=output_file_txt,
            filename="monitor_results.json",
            media_type="application/json"
        )

    def handle_ref_result_selection(self, e: events.TableSelectionEventArguments):
        # if selection is empty, clear the data
        if len(e.selection) == 0:
            self.ref_current_article = None
            self.ref_label_title.set_text("Title")
            self.ref_label_abstract.set_text("Abstract")
            self.ref_label_query.set_text("Query")
            self.ref_cb_article_relevant.set_value(False)
            self.ref_ta_article_eval.set_value("")
            return
        row_data = e.selection[0]
        # set current article
        self.ref_current_article = self.ref_agent_results.new_articles[row_data['index']]

        self.ref_label_title.set_text(row_data['title'])
        self.ref_label_abstract.set_text(row_data['abstract'])
        self.ref_label_query.set_text(row_data['query'])
        self.ref_cb_article_relevant.set_value(row_data['is_relevant'])
        self.ref_ta_article_eval.set_value(row_data['evaluation'])
    
    def handle_ref_result_update(self):
        """Called when result relevance or evaluation is updated."""
        # if nothing selected, skip
        if self.ref_current_article is None:
            return
        self.ref_current_article.is_relevant = self.ref_cb_article_relevant.value
        self.ref_current_article.evaluation = self.ref_ta_article_eval.value
        # search for article pmid in table to get index
        row = None
        for r in self.ref_table_results_data.rows:
            if r['pubmed_id'] == self.ref_current_article.pubmed_id:
                row = r
                break
        if row is None:
            print("Warning! Article not found!")
            return
        # pull row
        row['is_relevant'] = self.ref_current_article.is_relevant
        row['evaluation'] = self.ref_current_article.evaluation
        # change the data
        self.ref_table_results_data.update()
    
    def handle_ref_prompt_update(self):
        """Called when one of the agent prompts is updated."""
        self.ref_agent_results.agent_system_prompt = self.ref_ta_system_prompt.value
        self.ref_agent_results.article_relevance_prompt = self.ref_ta_relevance_prompt.value
    
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