from multiprocessing import freeze_support

import flet as ft
from flet import TextButton

from engine.Search import Search


def main(page: ft.Page):
    def search_action(value, search):
        lv.clean()
        corp = search.corpus_manager
        search = search.search(value)
        for index, ele, in enumerate(search.tf_idf_vectors.get("lemmatized")):
            if index == 10:
                break
            document = corp.get_document_by_id(ele[1])
            lv.controls.append(ft.Text(f"DOC ID: {document.metadata.doc_id} URL {document.metadata.url}"))
        lv.update()
    search = Search()
    lv = ft.ListView(expand=1, spacing=10, item_extent=50)
    search_input = ft.TextField(label="Search", width=300)
    search_button = TextButton("Search", on_click=lambda e: search_action(search_input.value, search))
    page.add(search_input, search_button)
    page.add(lv)


if __name__ == "__main__":
    ft.app(target=main)
