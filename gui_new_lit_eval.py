from nicegui import ui, events
from sklearn.metrics import confusion_matrix

import numpy as np
import pandas as pd
import io

#import eval_new_lit_monitor as em

# toggle to control light/dark theme
dark = ui.dark_mode(value=True)
ui.switch('Dark mode').bind_value(dark)

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

# file uploader to select the evaluation results we want to look at
eval_result_uploader = ui.upload(
    on_upload=handle_upload,
    max_file_size=10e6,
    multiple=False,
    max_files=1,
    auto_upload=True,
    label="Upload evaluation results:"
)
eval_result_uploader.props('accept=.csv')

# global variable to store the results being looked at
eval_results_data = None

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
    table_conf_mat = ui.table(rows=rows, columns=columns, row_key='row_label')

    # miscellaneous stats
    with ui.column():
        label_accuracy = ui.label("Accuracy:")
        label_ppv = ui.label("PPV:")
        label_npv = ui.label("NPV:")

# table to put the results in
columns = [
        {'name': 'title', 'label': 'Title', 'field': 'title', 'required': True, 'align': 'left'},
        {'name': 'date', 'label': 'Published', 'field': 'date', 'sortable': True},
        {'name': 'is_relevant', 'label': 'Relevant? (LLM)', 'field':'is_relevant'},
        {'name': 'gold_standard', 'label': 'Relevant? (Human)', 'field':'gold_standard'}
    ]

# lay out components
def handle_result_selection(e: events.TableSelectionEventArguments):
    row_data = e.selection[0]
    label_title.set_text(row_data['title'])
    label_abstract.set_text(row_data['abstract'])
    label_query.set_text(row_data['query'])
    label_llm_eval.set_text(row_data['evaluation'])

# table showing the articles in this evaluation run
table_results_data = ui.table(
    rows=[], 
    columns=columns,
    selection='single', 
    row_key='pubmed_id',
    pagination=3,
    on_select=handle_result_selection,
)

label_title = ui.label("Title").classes("text-3xl")
label_abstract = ui.label().classes("text-base")
# the criteria for article relevance
ui.label("Relevance criteria:").classes("text-2xl")
label_query = ui.label().classes("text-base")
# what the LLM thought about the article
ui.label("LLM evaluation:").classes("text-2xl")
label_llm_eval = ui.label().classes("text-base")

ui.label("Was LLM right or wrong? Why?").classes("text-2xl")
ta_error_disc = ui.textarea(
    placeholder="Write your explanation here.",
)

ui.run(host='127.0.0.1', port=9090, title="Lit Monitor Evaluator")