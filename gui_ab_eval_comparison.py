import numpy as np
from nicegui import events, ui
from nicegui.elements.checkbox import Checkbox
from nicegui.elements.label import Label
from nicegui.elements.table import Table
from nicegui.elements.textarea import Textarea
from pydantic import ValidationError

from monitor import Article, LitMonitorState


class ABEvalGUI:
    """
    A GUI for side-by-side comparison of evaluations run under two different conditions.
    """

    # ===== Reference Results =============
    ref_agent_results: LitMonitorState = None
    ref_current_article: Article
    """Currently selected article, if any."""

    # GUI elements
    ref_ta_system_prompt: Textarea
    """Text area showing the system prompt used by the agent."""
    ref_ta_relevance_prompt: Textarea
    """Text area showing the format used to present articles for evaluation."""

    ref_table_results_data: Table
    """Table showing articles in the current result."""
    ref_label_title: Label
    """Label to hold selected article title."""
    ref_label_abstract: Label
    """Label to hold selected article abstract."""
    ref_label_query: Label
    """Label to hold the criteria for whether an article is relevant."""
    ref_cb_article_relevant: Checkbox
    """Check box showing/controlling whether article is judged as relevant."""
    ref_ta_article_eval: Textarea
    """Text area showing the article relevance evaluation."""

    # ===== Condition A Results =============
    condA_agent_results: LitMonitorState = None
    condA_current_article: Article
    """Currently selected article, if any."""

    # GUI elements
    condA_table_conf_mat: Table
    """Table displaying confusion matrix for condition A."""
    condA_label_accuracy: Label
    condA_label_ppv: Label
    condA_label_npv: Label
    condA_ta_system_prompt: Textarea
    """Text area showing the system prompt used by the agent."""
    condA_ta_relevance_prompt: Textarea
    """Text area showing the format used to present articles for evaluation."""

    condA_table_results_data: Table
    """Table showing articles in the current result."""
    condA_label_title: Label
    """Label to hold selected article title."""
    condA_label_abstract: Label
    """Label to hold selected article abstract."""
    condA_label_query: Label
    """Label to hold the criteria for whether an article is relevant."""
    condA_cb_article_relevant: Checkbox
    """Check box showing/controlling whether article is judged as relevant."""
    condA_ta_article_eval: Textarea
    """Text area showing the article relevance evaluation."""
    condA_ta_article_ref_eval: Textarea
    """Text area showing the article's gold-standard reference relevance evaluation."""

    # ===== Condition B Results =============
    condB_agent_results: LitMonitorState = None
    condB_current_article: Article
    """Currently selected article, if any."""

    # GUI elements
    condB_table_conf_mat: Table
    """Table displaying confusion matrix for condition B."""
    condB_label_accuracy: Label
    condB_label_ppv: Label
    condB_label_npv: Label
    condB_ta_system_prompt: Textarea
    """Text area showing the system prompt used by the agent."""
    condB_ta_relevance_prompt: Textarea
    """Text area showing the format used to present articles for evaluation."""

    condB_table_results_data: Table
    """Table showing articles in the current result."""
    condB_label_title: Label
    """Label to hold selected article title."""
    condB_label_abstract: Label
    """Label to hold selected article abstract."""
    condB_label_query: Label
    """Label to hold the criteria for whether an article is relevant."""
    condB_cb_article_relevant: Checkbox
    """Check box showing/controlling whether article is judged as relevant."""
    condB_ta_article_eval: Textarea
    """Text area showing the article relevance evaluation."""
    condB_ta_article_ref_eval: Textarea
    """Text area showing the article's gold-standard reference relevance evaluation."""

    comp_label_mcnemar: Label
    """Label for displaying the McNemar's test p-value comparing conditions A and B."""
    comp_table_conf_mat: Table
    """Table displaying confusion matrix for condition A vs. condition B."""

    def __init__(self):
        """Initialize the GUI."""
        # set up state
        # set up GUI
        self.dark_setting = ui.dark_mode(value=True)
        self.setup_ui()

    def setup_ui(self):
        """Build the GUI itself."""
        # define navigation tabs
        with (
            ui.header().classes('bg-dark'),
            ui.tabs().classes('w-full') as tabs
        ):
            tab_ref = ui.tab('Reference')
            tab_cond_a = ui.tab('Condition A')
            tab_cond_b = ui.tab('Condition B')
            tab_comparison = ui.tab("Comparison")
        # define contents of each tab
        with ui.tab_panels(tabs, value=tab_ref).classes('w-7/8'):

            # ------ REFERENCE TAB -------------

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
                    {'name': 'title', 'label': 'Title', 'field': 'title', 'required': True, 'align': 'left', 'style': 'text-wrap: wrap'},
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

            # ------ CONDITION A TAB -------------

            condA_panel = ui.tab_panel(tab_cond_a)
            with condA_panel:
                with ui.row():
                    # file uploader to select the evaluation results we want to look at
                    condA_eval_result_uploader = ui.upload(
                        on_upload=self.handle_condA_upload,
                        max_file_size=10e6,
                        multiple=False,
                        max_files=1,
                        auto_upload=True,
                        label="Upload evaluation results:"
                    )
                    condA_eval_result_uploader.props('accept=.json')

                    ui.button(text="Save", icon='save', on_click=self.handle_condA_save)
                
                # display summary statistics about the results

                with ui.row():
                    # confusion matrix
                    columns = [
                        {'name': 'row_label', 'label': 'True relevance', 'field': 'row_label'},
                        {'name': 'negative', 'label': 'Pred. irrelevant', 'field': 'negative'},
                        {'name': 'positive', 'label': 'Pred. relevant', 'field': 'positive'},
                    ]
                    # placeholder data
                    rows = [
                        {'row_label': 'Irrelevant', 'positive': 0, 'negative': 0},
                        {'row_label': 'Relevant', 'positive': 0, 'negative': 0},
                    ]
                    self.condA_table_conf_mat = ui.table(rows=rows, columns=columns, row_key='row_label')

                    # miscellaneous stats
                    with ui.column():
                        self.condA_label_accuracy = ui.label("Accuracy:")
                        self.condA_label_ppv = ui.label("PPV:")
                        self.condA_label_npv = ui.label("NPV:")

                # common settings for all articles

                ui.label("System prompt:").classes("text-2xl")
                self.condA_ta_system_prompt = ui.textarea(
                    placeholder="Agent's system prompt.",
                    on_change=self.handle_condA_prompt_update
                ).classes("text-base w-7/8")

                ui.label("Article relevance prompt:").classes("text-2xl")
                self.condA_ta_relevance_prompt = ui.textarea(
                    placeholder="Prompt used to present articles for evaluation.",
                    on_change=self.handle_condA_prompt_update
                ).classes("text-base w-7/8")

                # table showing the articles in this evaluation run
                columns = [
                    {'name': 'title', 'label': 'Title', 'field': 'title', 'required': True, 'align': 'left', 'style': 'text-wrap: wrap'},
                    {'name': 'date', 'label': 'Published', 'field': 'date', 'sortable': True},
                    {'name': 'is_relevant', 'label': 'Predicted Relevant', 'field':'is_relevant', 'sortable': True},
                    {'name': 'ref_is_relevant', 'label': 'Actually Relevant', 'field':'ref_is_relevant', 'sortable': True}
                ]
                self.condA_table_results_data = ui.table(
                    rows=[], 
                    columns=columns,
                    selection='single', 
                    row_key='pubmed_id',
                    pagination=3,
                    on_select=self.handle_condA_result_selection,
                ).classes("w-7/8")

                # information about an individual article

                self.condA_label_title = ui.label("Title").classes("text-3xl w-7/8")
                self.condA_label_abstract = ui.label().classes("text-base w-7/8")
                # the criteria for article relevance
                ui.label("Relevance criteria:").classes("text-2xl")
                self.condA_label_query = ui.label().classes("text-base w-7/8")
                # what the LLM thought about the article

                self.condA_cb_article_relevant = ui.checkbox(
                    text="Article relevant to query",
                    value=False,
                    on_change=self.handle_condA_result_update
                )
                ui.label("Why is/isn't the article relevant?").classes("text-2xl")
                with ui.row(align_items="center").classes("w-full"):
                    # first row, with labels
                    ui.label("Proposed explanation:").classes("w-3/8")
                    ui.label("True explanation:").classes("w-3/8")
                    # second row, with text areas
                    self.condA_ta_article_eval = ui.textarea(
                        placeholder="Write explanation here.",
                        on_change=self.handle_condA_result_update
                    ).classes("text-base w-3/8")
                    self.condA_ta_article_ref_eval = ui.textarea(
                        placeholder="Reference explanation here."
                    ).classes("text-base w-3/8")
                    self.condA_ta_article_ref_eval.disable()

            # ------ CONDITION B TAB -------------

            condB_panel = ui.tab_panel(tab_cond_b)
            with condB_panel:
                with ui.row():
                    # file uploader to select the evaluation results we want to look at
                    condB_eval_result_uploader = ui.upload(
                        on_upload=self.handle_condB_upload,
                        max_file_size=10e6,
                        multiple=False,
                        max_files=1,
                        auto_upload=True,
                        label="Upload evaluation results:"
                    )
                    condB_eval_result_uploader.props('accept=.json')

                    ui.button(text="Save", icon='save', on_click=self.handle_condB_save)
                
                # display summary statistics about the results

                with ui.row():
                    # confusion matrix
                    columns = [
                        {'name': 'row_label', 'label': 'True relevance', 'field': 'row_label'},
                        {'name': 'negative', 'label': 'Pred. irrelevant', 'field': 'negative'},
                        {'name': 'positive', 'label': 'Pred. relevant', 'field': 'positive'},
                    ]
                    # placeholder data
                    rows = [
                        {'row_label': 'Irrelevant', 'positive': 0, 'negative': 0},
                        {'row_label': 'Relevant', 'positive': 0, 'negative': 0},
                    ]
                    self.condB_table_conf_mat = ui.table(rows=rows, columns=columns, row_key='row_label')

                    # miscellaneous stats
                    with ui.column():
                        self.condB_label_accuracy = ui.label("Accuracy:")
                        self.condB_label_ppv = ui.label("PPV:")
                        self.condB_label_npv = ui.label("NPV:")
                
                # common settings for all articles

                ui.label("System prompt:").classes("text-2xl")
                self.condB_ta_system_prompt = ui.textarea(
                    placeholder="Agent's system prompt.",
                    on_change=self.handle_condB_prompt_update
                ).classes("text-base w-7/8")

                ui.label("Article relevance prompt:").classes("text-2xl")
                self.condB_ta_relevance_prompt = ui.textarea(
                    placeholder="Prompt used to present articles for evaluation.",
                    on_change=self.handle_condB_prompt_update
                ).classes("text-base w-7/8")

                # table showing the articles in this evaluation run
                columns = [
                    {'name': 'title', 'label': 'Title', 'field': 'title', 'required': True, 'align': 'left', 'style': 'text-wrap: wrap'},
                    {'name': 'date', 'label': 'Published', 'field': 'date', 'sortable': True},
                    {'name': 'is_relevant', 'label': 'Predicted Relevant', 'field':'is_relevant', 'sortable': True},
                    {'name': 'ref_is_relevant', 'label': 'Actually Relevant', 'field':'ref_is_relevant', 'sortable': True}
                ]
                self.condB_table_results_data = ui.table(
                    rows=[], 
                    columns=columns,
                    selection='single', 
                    row_key='pubmed_id',
                    pagination=3,
                    on_select=self.handle_condB_result_selection,
                ).classes("w-7/8")

                # information about an individual article

                self.condB_label_title = ui.label("Title").classes("text-3xl w-7/8")
                self.condB_label_abstract = ui.label().classes("text-base w-7/8")
                # the criteria for article relevance
                ui.label("Relevance criteria:").classes("text-2xl")
                self.condB_label_query = ui.label().classes("text-base w-7/8")
                # what the LLM thought about the article

                self.condB_cb_article_relevant = ui.checkbox(
                    text="Article relevant to query",
                    value=False,
                    on_change=self.handle_condB_result_update
                )
                ui.label("Why is/isn't the article relevant?").classes("text-2xl")
                with ui.row(align_items="center").classes("w-full"):
                    # first row, with labels
                    ui.label("Proposed explanation:").classes("w-3/8")
                    ui.label("True explanation:").classes("w-3/8")
                    # second row, with text areas
                    self.condB_ta_article_eval = ui.textarea(
                        placeholder="Write explanation here.",
                        on_change=self.handle_condB_result_update
                    ).classes("text-base w-3/8")
                    self.condB_ta_article_ref_eval = ui.textarea(
                        placeholder="Reference explanation here."
                    ).classes("text-base w-3/8")
                    self.condB_ta_article_ref_eval.disable()

            # ------ COMPARISON TAB -------------

            comp_panel = ui.tab_panel(tab_comparison)
            with comp_panel:
                ui.label("Conditions significantly different? (McNemar's test)")
                self.comp_label_mcnemar = ui.label("")
                
                # comparison matrix
                columns = [
                    {'name': 'row_label', 'label': 'Cond. A Pred.', 'field': 'row_label'},
                    {'name': 'negative', 'label': 'Cond. B Pred.\nirrelevant', 'field': 'negative'},
                    {'name': 'positive', 'label': 'Cond B. Pred.\nrelevant', 'field': 'positive'},
                ]
                # placeholder data
                rows = [
                    {'row_label': 'Irrelevant', 'positive': 0, 'negative': 0},
                    {'row_label': 'Relevant', 'positive': 0, 'negative': 0},
                ]
                self.comp_table_conf_mat = ui.table(rows=rows, columns=columns, row_key='row_label')
    
    # ============ REFERENCE TAB EVENT HANDLERS ================

    async def handle_ref_upload(self, e: events.UploadEventArguments):
        """
        Uploads an agent result file and loads the data into the reference results table.

        Args:
            e: The file upload event.
        """
        # Read the result file
        try:
            text = await e.file.text()
            self.ref_agent_results = LitMonitorState.model_validate_json(json_data=text)
        except ValidationError as err:
            ui.notify(
                message=f"Error reading reference file:\n{err}",
                type='warning',
                multi_line=True
            )
            print(f"Error reading reference file:\n{e}")
            return
        
        # clear the upload widget
        e.sender.reset()

        # load prompts
        self.ref_ta_system_prompt.value = self.ref_agent_results.agent_system_prompt
        self.ref_ta_relevance_prompt.value = self.ref_agent_results.article_relevance_prompt
        
        # populate the table
        result_rows = []
        for index, article in enumerate(self.ref_agent_results.new_articles):
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
        self.ref_table_results_data.rows = result_rows
        self.update_all_comparisons()

    def handle_ref_save(self):
        """Save the monitor results to a JSON file."""
        if self.ref_agent_results is None:
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
        self.update_all_comparisons()
    
    def handle_ref_prompt_update(self):
        """Called when one of the agent prompts is updated."""
        self.ref_agent_results.agent_system_prompt = self.ref_ta_system_prompt.value
        self.ref_agent_results.article_relevance_prompt = self.ref_ta_relevance_prompt.value
    
    # ============ CONDITION A TAB EVENT HANDLERS ================
    
    async def handle_condA_upload(self, e: events.UploadEventArguments):
        """
        Uploads an agent result file and loads the data into the reference results table.

        Args:
            e: The file upload event.
        """
        # Read the result file
        try:
            text = await e.file.text()
            self.condA_agent_results = LitMonitorState.model_validate_json(json_data=text)
        except ValidationError as err:
            ui.notify(
                message=f"Error reading condition A file:\n{err}",
                type='warning',
                multi_line=True
            )
            print(f"Error reading condition A file:\n{e}")
            return
        
        # clear the upload widget
        e.sender.reset()

        # load prompts
        self.condA_ta_system_prompt.value = self.condA_agent_results.agent_system_prompt
        self.condA_ta_relevance_prompt.value = self.condA_agent_results.article_relevance_prompt
        
        # populate the table
        result_rows = []
        ref_matches = 0
        for index, article in enumerate(self.condA_agent_results.new_articles):
            if self.ref_agent_results is not None:
                ref_article = self.ref_agent_results.get_article_with_pubmed_id(article.pubmed_id)
            else:
                ref_article = None
            
            if ref_article is not None:
                ref_matches += 1
                ref_relevant = ref_article.is_relevant
                ref_eval = ref_article.evaluation
            else:
                ref_relevant = None
                ref_eval = ""

            row_data = {
                "index": index,
                "pubmed_id": article.pubmed_id,
                "date": article.date,
                "title": article.title,
                "source": article.source,
                "is_relevant": article.is_relevant,
                "ref_is_relevant": ref_relevant,
                "abstract": article.abstract,
                "query": self.condA_agent_results.topic_description,
                "evaluation": article.evaluation,
                "ref_evaluation": ref_eval
            }
            result_rows.append(row_data)
        self.condA_table_results_data.rows = result_rows
        self.update_all_comparisons()

    def handle_condA_save(self):
        """Save the monitor results to a JSON file."""
        if self.agent_results is None:
            return
        output_file_txt = self.condA_agent_results.model_dump_json(indent=2)
        # show save dialog
        ui.download.content(
            content=output_file_txt,
            filename="monitor_results.json",
            media_type="application/json"
        )

    def handle_condA_result_selection(self, e: events.TableSelectionEventArguments):
        # if selection is empty, clear the data
        if len(e.selection) == 0:
            self.condA_current_article = None
            self.condA_label_title.set_text("Title")
            self.condA_label_abstract.set_text("Abstract")
            self.condA_label_query.set_text("Query")
            self.condA_cb_article_relevant.set_value(False)
            self.condA_ta_article_eval.set_value("")
            self.condA_ta_article_ref_eval.set_value("")
            return
        row_data = e.selection[0]
        # set current article
        self.condA_current_article = self.condA_agent_results.new_articles[row_data['index']]

        self.condA_label_title.set_text(row_data['title'])
        self.condA_label_abstract.set_text(row_data['abstract'])
        self.condA_label_query.set_text(row_data['query'])
        self.condA_cb_article_relevant.set_value(row_data['is_relevant'])
        self.condA_ta_article_eval.set_value(row_data['evaluation'])
        self.condA_ta_article_ref_eval.set_value(row_data['ref_evaluation'])
    
    def handle_condA_result_update(self):
        """Called when result relevance or evaluation is updated."""
        # if nothing selected, skip
        if self.condA_current_article is None:
            return
        self.condA_current_article.is_relevant = self.condA_cb_article_relevant.value
        self.condA_current_article.evaluation = self.condA_ta_article_eval.value
        # search for article pmid in table to get index
        row = None
        for r in self.condA_table_results_data.rows:
            if r['pubmed_id'] == self.condA_current_article.pubmed_id:
                row = r
                break
        if row is None:
            print("Warning! Article not found!")
            return
        # pull row
        row['is_relevant'] = self.condA_current_article.is_relevant
        row['evaluation'] = self.condA_current_article.evaluation
        # change the data
        self.condA_table_results_data.update()
        self.update_all_comparisons()
    
    def handle_condA_prompt_update(self):
        """Called when one of the agent prompts is updated."""
        self.condA_agent_results.agent_system_prompt = self.condA_ta_system_prompt.value
        self.condA_agent_results.article_relevance_prompt = self.condA_ta_relevance_prompt.value
    
    # ============ CONDITION B TAB EVENT HANDLERS ================

    async def handle_condB_upload(self, e: events.UploadEventArguments):
        """
        Uploads an agent result file and loads the data into the reference results table.

        Args:
            e: The file upload event.
        """
        # Read the result file
        try:
            text = await e.file.text()
            self.condB_agent_results = LitMonitorState.model_validate_json(json_data=text)
        except ValidationError as err:
            ui.notify(
                message=f"Error reading condition B file:\n{err}",
                type='warning',
                multi_line=True
            )
            print(f"Error reading condition B file:\n{e}")
            return
        
        # clear the upload widget
        e.sender.reset()

        # load prompts
        self.condB_ta_system_prompt.value = self.condB_agent_results.agent_system_prompt
        self.condB_ta_relevance_prompt.value = self.condB_agent_results.article_relevance_prompt
        
        # populate the table
        result_rows = []
        ref_matches = 0
        for index, article in enumerate(self.condB_agent_results.new_articles):
            if self.ref_agent_results is not None:
                ref_article = self.ref_agent_results.get_article_with_pubmed_id(article.pubmed_id)
            else:
                ref_article = None
            
            if ref_article is not None:
                ref_matches += 1
                ref_relevant = ref_article.is_relevant
                ref_eval = ref_article.evaluation
            else:
                ref_relevant = None
                ref_eval = ""

            row_data = {
                "index": index,
                "pubmed_id": article.pubmed_id,
                "date": article.date,
                "title": article.title,
                "source": article.source,
                "is_relevant": article.is_relevant,
                "ref_is_relevant": ref_relevant,
                "abstract": article.abstract,
                "query": self.condB_agent_results.topic_description,
                "evaluation": article.evaluation,
                "ref_evaluation": ref_eval
            }
            result_rows.append(row_data)
        self.condB_table_results_data.rows = result_rows
        self.update_all_comparisons()

    def handle_condB_save(self):
        """Save the monitor results to a JSON file."""
        if self.agent_results is None:
            return
        output_file_txt = self.condB_agent_results.model_dump_json(indent=2)
        # show save dialog
        ui.download.content(
            content=output_file_txt,
            filename="monitor_results.json",
            media_type="application/json"
        )

    def handle_condB_result_selection(self, e: events.TableSelectionEventArguments):
        # if selection is empty, clear the data
        if len(e.selection) == 0:
            self.condB_current_article = None
            self.condB_label_title.set_text("Title")
            self.condB_label_abstract.set_text("Abstract")
            self.condB_label_query.set_text("Query")
            self.condB_cb_article_relevant.set_value(False)
            self.condB_ta_article_eval.set_value("")
            self.condA_ta_article_ref_eval.set_value("")
            return
        row_data = e.selection[0]
        # set current article
        self.condB_current_article = self.condB_agent_results.new_articles[row_data['index']]

        self.condB_label_title.set_text(row_data['title'])
        self.condB_label_abstract.set_text(row_data['abstract'])
        self.condB_label_query.set_text(row_data['query'])
        self.condB_cb_article_relevant.set_value(row_data['is_relevant'])
        self.condB_ta_article_eval.set_value(row_data['evaluation'])
        self.condB_ta_article_ref_eval.set_value(row_data['ref_evaluation'])
    
    def handle_condB_result_update(self):
        """Called when result relevance or evaluation is updated."""
        # if nothing selected, skip
        if self.condB_current_article is None:
            return
        self.condB_current_article.is_relevant = self.condB_cb_article_relevant.value
        self.condB_current_article.evaluation = self.condB_ta_article_eval.value
        # search for article pmid in table to get index
        row = None
        for r in self.condB_table_results_data.rows:
            if r['pubmed_id'] == self.condB_current_article.pubmed_id:
                row = r
                break
        if row is None:
            print("Warning! Article not found!")
            return
        # pull row
        row['is_relevant'] = self.condB_current_article.is_relevant
        row['evaluation'] = self.condB_current_article.evaluation
        # change the data
        self.condB_table_results_data.update()
        self.update_all_comparisons()
    
    def handle_condB_prompt_update(self):
        """Called when one of the agent prompts is updated."""
        self.condB_agent_results.agent_system_prompt = self.condB_ta_system_prompt.value
        self.condB_agent_results.article_relevance_prompt = self.condB_ta_relevance_prompt.value
    
    def update_all_comparisons(self):
        """Checks to see if article lists match and updates comparison statistics."""
        # compare reference and condition A
        if self.ref_agent_results is not None and self.condA_agent_results is not None:
            ref_matches = 0
            rel_true = []
            rel_condA = []
            for article in self.condA_agent_results.new_articles:
                ref_article = self.ref_agent_results.get_article_with_pubmed_id(article.pubmed_id)
                if ref_article is not None:
                    ref_matches += 1
                    rel_true.append(ref_article.is_relevant)
                    rel_condA.append(article.is_relevant)
            if ref_matches < len(self.ref_agent_results.new_articles):
                ui.notify(
                    message="Not all reference articles present in condition A!",
                    type='warning'
                )

            # update condition A statistics
            
            y_true = np.array(rel_true, dtype=np.bool)
            y_pred = np.array(rel_condA, dtype=np.bool)
            cm = self._confusion_matrix(
                ref_data=y_true,
                predicted_data=y_pred
            )
            self.condA_table_conf_mat.rows = [
                {'row_label': 'Irrelevant', 'negative': cm[0, 0], 'positive': cm[0, 1]},
                {'row_label': 'Relevant', 'negative': cm[1, 0], 'positive': cm[1, 1]},
            ]

            # calculate accuracy
            pred_accuracy = np.sum(y_true == y_pred)/len(y_true)
            self.condA_label_accuracy.text = f"Accuracy: {pred_accuracy:.1%}"
            # calculate PPV
            true_positives = np.sum((y_true == 1) & (y_pred == 1))
            false_positives = np.sum((y_true == 0) & (y_pred == 1))
            if (true_positives + false_positives) > 0:
                ppv = true_positives/(true_positives + false_positives)
            else:
                ppv = 0.0
            self.condA_label_ppv.text = f"PPV: {ppv:.1%}"
            # calculate NPV
            true_negatives = np.sum((y_true == 0) & (y_pred == 0))
            false_negatives = np.sum((y_true == 1) & (y_pred == 0))
            if (true_negatives + false_negatives) > 0:
                npv = true_negatives/(true_negatives + false_negatives)
            else:
                npv = 0.0
            self.condA_label_npv.text = f"NPV: {npv:.1%}"

        # compare reference and condition B
        if self.ref_agent_results is not None and self.condB_agent_results is not None:
            ref_matches = 0
            rel_true = []
            rel_condB = []
            for article in self.condB_agent_results.new_articles:
                ref_article = self.ref_agent_results.get_article_with_pubmed_id(article.pubmed_id)
                if ref_article is not None:
                    ref_matches += 1
                    rel_true.append(ref_article.is_relevant)
                    rel_condB.append(article.is_relevant)
            if ref_matches < len(self.ref_agent_results.new_articles):
                ui.notify(
                    message="Not all reference articles present in condition B!",
                    type='warning'
                )

            # update condition B statistics
            
            y_true = np.array(rel_true, dtype=np.bool)
            y_pred = np.array(rel_condB, dtype=np.bool)
            cm = self._confusion_matrix(
                ref_data=y_true,
                predicted_data=y_pred
            )
            self.condB_table_conf_mat.rows = [
                {'row_label': 'Irrelevant', 'negative': cm[0, 0], 'positive': cm[0, 1]},
                {'row_label': 'Relevant', 'negative': cm[1, 0], 'positive': cm[1, 1]},
            ]

            # calculate accuracy
            pred_accuracy = np.sum(y_true == y_pred)/len(y_true)
            self.condB_label_accuracy.text = f"Accuracy: {pred_accuracy:.1%}"
            # calculate PPV
            true_positives = np.sum((y_true == 1) & (y_pred == 1))
            false_positives = np.sum((y_true == 0) & (y_pred == 1))
            if (true_positives + false_positives) > 0:
                ppv = true_positives/(true_positives + false_positives)
            else:
                ppv = 0.0
            self.condB_label_ppv.text = f"PPV: {ppv:.1%}"
            # calculate NPV
            true_negatives = np.sum((y_true == 0) & (y_pred == 0))
            false_negatives = np.sum((y_true == 1) & (y_pred == 0))
            if (true_negatives + false_negatives) > 0:
                npv = true_negatives/(true_negatives + false_negatives)
            else:
                npv = 0.0
            self.condB_label_npv.text = f"NPV: {npv:.1%}"

        # compare condition A and condition B
        if self.condA_agent_results is not None and self.condB_agent_results is not None:
            ref_matches = 0
            rel_condA = []
            rel_condB = []
            for article in self.condB_agent_results.new_articles:
                ref_article = self.condA_agent_results.get_article_with_pubmed_id(article.pubmed_id)
                if ref_article is not None:
                    ref_matches += 1
                    rel_condA.append(ref_article.is_relevant)
                    rel_condB.append(article.is_relevant)
            if ref_matches < len(self.condA_agent_results.new_articles):
                ui.notify(
                    message="Not all condition A articles present in condition B!",
                    type='warning'
                )
            # update comparison statistics here!
            y_true = np.array(rel_condA, dtype=np.bool)
            y_pred = np.array(rel_condB, dtype=np.bool)
            comp_mat = self._confusion_matrix(
                ref_data=y_true,
                predicted_data=y_pred
            )
            mn_stat, p_value = self._mcnemar(
                table=comp_mat,
                exact=False
            )
            self.comp_label_mcnemar.text = f"p-value: {p_value:.4}; statistic: {mn_stat:.4}"

            self.comp_table_conf_mat.rows = [
                {'row_label': 'Irrelevant', 'negative': comp_mat[0, 0], 'positive': comp_mat[0, 1]},
                {'row_label': 'Relevant', 'negative': comp_mat[1, 0], 'positive': comp_mat[1, 1]},
            ]
    
    def _confusion_matrix(self, ref_data, predicted_data) -> np.array:
        if len(ref_data) != len(predicted_data):
            raise ValueError("Can't call confusion_matrix with arrays of different lengths!")
        conf_mat = np.zeros((2, 2), dtype=np.int_)

        # Ensure inputs are numpy arrays
        y_true = np.asarray(ref_data, dtype=bool)
        y_pred = np.asarray(predicted_data, dtype=bool)

        np.sum((y_true == 1) & (y_pred == 1))

        tp = np.sum((y_true == 1) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))

        conf_mat = np.array([
            [tn, fp],
            [fn, tp]
        ])
        return conf_mat

    def _mcnemar(self, table, exact=False):
        """
        Perform McNemar's test using NumPy.
        
        Parameters
        ----------
        table : array-like of shape (2, 2)
            Contingency table:
                [[n00, n01],
                [n10, n11]]
        exact : bool
            If True, compute exact binomial test p-value.
            If False, compute chi-square test with continuity correction.
            
        Returns
        -------
        statistic : float
            Test statistic (chi-square or binomial test statistic).
        p_value : float
            Corresponding p-value.
        """
        table = np.asarray(table)
        if table.shape != (2, 2):
            raise ValueError("Input table must be 2x2.")

        b = table[0, 1]
        c = table[1, 0]

        # Chi-square version (with continuity correction)
        if not exact:
            if b + c == 0:
                return np.nan, 1.0
            statistic = (abs(b - c) - 1)**2 / (b + c)
            # p-value from chi-square(1 df)
            # CDF = 1 - exp(-x/2)
            p_value = np.exp(-statistic / 2)
            return statistic, p_value

        # Exact binomial test
        n = b + c
        if n == 0:
            return np.nan, 1.0

        # Compute binomial CDF using NumPy
        # CDF(k; n, 0.5) = sum_{i=0..k} binom(n, i) * 0.5^n
        ks = np.arange(0, b + 1)
        # log binomial coefficients via gammaln
        log_binom = np.gammaln(n + 1) - np.gammaln(ks + 1) - np.gammaln(n - ks + 1)
        cdf_b = np.sum(np.exp(log_binom - n * np.log(2)))

        p_value = 2 * min(cdf_b, 1 - cdf_b)
        return None, p_value

# gui = ABEvalGUI()
# ui.run(host='127.0.0.1', port=9092, title="New Lit A/B Eval")

# wrapper function so every user session gets its own UI object
def main():
    eval_ui = ABEvalGUI()
    eval_ui.setup_ui()

if __name__ in {"__main__", "__mp_main__"}:
    ui.run(root=main, host='127.0.0.1', port=9092, title="New Lit A/B Eval", favicon='🥔',
        binding_refresh_interval=0.2, reconnect_timeout=10
    )