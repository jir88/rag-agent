from nicegui import ui, events, elements

import sys

from monitor import Article,LitMonitorState

class EvalGUI:
    """Class managing the literature evaluation GUI."""

    agent_results: LitMonitorState
    current_article: Article
    """Currently selected article, if any."""

    # GUI elements
    ta_system_prompt: elements.textarea.Textarea
    """Text area showing the system prompt used by the agent."""
    ta_relevance_prompt: elements.textarea.Textarea
    """Text area showing the format used to present articles for evaluation."""

    table_results_data: elements.table.Table
    """Table showing articles in the current result."""
    label_title: elements.label.Label
    """Label to hold selected article title."""
    label_abstract: elements.label.Label
    """Label to hold selected article abstract."""
    label_query: elements.label.Label
    """Label to hold the criteria for whether an article is relevant."""
    cb_article_relevant: elements.checkbox.Checkbox
    """Check box showing/controlling whether article is judged as relevant."""
    ta_article_eval: elements.textarea.Textarea
    """Text area showing the article relevance evaluation."""

    def __init__(self):
        self.agent_results: LitMonitorState = None
        self.current_article: Article = None

        # initialize GUI

        # toggle to control light/dark theme
        dark = ui.dark_mode(value=True)
        ui.switch('Dark mode').bind_value(dark)

        with ui.row():
            # file uploader to select the evaluation results we want to look at
            eval_result_uploader = ui.upload(
                on_upload=self.handle_upload,
                max_file_size=10e6,
                multiple=False,
                max_files=1,
                auto_upload=True,
                label="Upload evaluation results:"
            )
            eval_result_uploader.props('accept=.json')

            ui.button(text="Save", icon='save', on_click=self.handle_save)
        
        # common settings for all articles

        ui.label("System prompt:").classes("text-2xl")
        self.ta_system_prompt = ui.textarea(
            placeholder="Agent's system prompt.",
            on_change=self.handle_prompt_update
        ).classes("text-base w-7/8")

        ui.label("Article relevance prompt:").classes("text-2xl")
        self.ta_relevance_prompt = ui.textarea(
            placeholder="Prompt used to present articles for evaluation.",
            on_change=self.handle_prompt_update
        ).classes("text-base w-7/8")

        # table showing the articles in this evaluation run
        columns = [
            {'name': 'title', 'label': 'Title', 'field': 'title', 'required': True, 'align': 'left'},
            {'name': 'date', 'label': 'Published', 'field': 'date', 'sortable': True},
            {'name': 'is_relevant', 'label': 'Relevant?', 'field':'is_relevant', 'sortable': True}
        ]
        self.table_results_data = ui.table(
            rows=[], 
            columns=columns,
            selection='single', 
            row_key='pubmed_id',
            pagination=3,
            on_select=self.handle_result_selection,
        ).classes("w-7/8")

        # information about an individual article

        self.label_title = ui.label("Title").classes("text-3xl w-7/8")
        self.label_abstract = ui.label().classes("text-base w-7/8")
        # the criteria for article relevance
        ui.label("Relevance criteria:").classes("text-2xl")
        self.label_query = ui.label().classes("text-base w-7/8")
        # what the LLM thought about the article

        self.cb_article_relevant = ui.checkbox(
            text="Article relevant to query",
            value=False,
            on_change=self.handle_result_update
        )
        ui.label("Why is/isn't the article relevant?").classes("text-2xl")
        self.ta_article_eval = ui.textarea(
            placeholder="Write explanation here.",
            on_change=self.handle_result_update
        ).classes("text-base w-7/8")

    async def handle_upload(self, e: events.UploadEventArguments):
        """
        Uploads an agent result file and loads the data into the table.

        Args:
            e: The file upload event.
        """
        # Read the result file
        try:
            self.agent_results = LitMonitorState.model_validate_json(await e.file.text())
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
        self.ta_system_prompt.value = self.agent_results.agent_system_prompt
        self.ta_relevance_prompt.value = self.agent_results.article_relevance_prompt
        
        # populate the table
        result_rows = []
        index = 0
        for article in self.agent_results.new_articles:
            row_data = {
                "index": index,
                "pubmed_id": article.pubmed_id,
                "date": article.date,
                "title": article.title,
                "source": article.source,
                "is_relevant": article.is_relevant,
                "abstract": article.abstract,
                "query": self.agent_results.topic_description,
                "evaluation": article.evaluation
            }
            result_rows.append(row_data)
            index += 1
        self.table_results_data.rows = result_rows
    
    def handle_save(self):
        """Save the monitor results to a JSON file."""
        if self.agent_results is None:
            return
        output_file_txt = self.agent_results.model_dump_json(indent=2)
        # show save dialog
        ui.download.content(
            content=output_file_txt,
            filename="monitor_results.json",
            media_type="application/json"
        )

    def handle_result_selection(self, e: events.TableSelectionEventArguments):
        # if selection is empty, clear the data
        if len(e.selection) == 0:
            self.current_article = None
            self.label_title.set_text("Title")
            self.label_abstract.set_text("Abstract")
            self.label_query.set_text("Query")
            self.cb_article_relevant.set_value(False)
            self.ta_article_eval.set_value("")
            return
        row_data = e.selection[0]
        # set current article
        self.current_article = self.agent_results.new_articles[row_data['index']]

        self.label_title.set_text(row_data['title'])
        self.label_abstract.set_text(row_data['abstract'])
        self.label_query.set_text(row_data['query'])
        self.cb_article_relevant.set_value(row_data['is_relevant'])
        self.ta_article_eval.set_value(row_data['evaluation'])
    
    def handle_result_update(self):
        """Called when result relevance or evaluation is updated."""
        # if nothing selected, skip
        if self.current_article is None:
            return
        self.current_article.is_relevant = self.cb_article_relevant.value
        self.current_article.evaluation = self.ta_article_eval.value
        # TODO: update the table too!
        # search for article pmid in table to get index
        row = None
        for r in self.table_results_data.rows:
            if r['pubmed_id'] == self.current_article.pubmed_id:
                row = r
                break
        if row is None:
            print("Warning! Article not found!")
            return
        # pull row
        row['is_relevant'] = self.current_article.is_relevant
        row['evaluation'] = self.current_article.evaluation
        # change the data
        self.table_results_data.update()
        print(self.agent_results.model_dump_json(indent=2))
    
    def handle_prompt_update(self):
        """Called when one of the agent prompts is updated."""
        self.agent_results.agent_system_prompt = self.ta_system_prompt.value
        self.agent_results.article_relevance_prompt = self.ta_relevance_prompt.value

# wrapper function so every user session gets its own UI object
async def main():
    EvalGUI()

if __name__ in {"__main__", "__mp_main__"}:
    ui.run(root=main, host='127.0.0.1', port=9090, title="Lit Monitor Evaluator",
        binding_refresh_interval=0.2, reconnect_timeout=10
    )