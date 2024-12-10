from nicegui import ui


#                           'table {width:100%;table-layout:fixed;} '
#                           'td {border:1px solid;padding: 8px;} '
#                           'dt {padding: 0;line-height: 1.0;} '
#                           'dd {padding: 0;line-height: 1.5;margin-left: 30px} '
#                           'th {border: 1px solid; padding: 2px;} '

# noinspection PyPep8Naming
class table(ui.element):
    def __init__(self, table_classes: str = None):
        super().__init__('table')
        tclasses = 'border-solid border border-black w-11/12 table-fixed'
        self.classes(tclasses if table_classes is None else table_classes)


# noinspection PyPep8Naming
class tr(ui.element):
    def __init__(self, tr_classes: str = None):
        super().__init__('tr')
        trclasses = ''
        self.classes(trclasses if tr_classes is None else tr_classes)


# noinspection PyPep8Naming
class th(ui.element):
    th_base_classes = 'font-bold border-solid border border-black p-2'
    label_base_classes = 'text-base'

    def __init__(self, label: str, th_classes: str = None, label_classes: str = None):
        super().__init__('th')
        self.classes(self.th_base_classes if th_classes is None else th_classes)
        with self:
            ui.label(label).classes(self.label_base_classes if label_classes is None else label_classes)


# noinspection PyPep8Naming
class td(ui.element):
    td_base_classes = 'border-solid border border-black p-2'
    label_base_classes = 'text-base'
    tt_base_classes = 'bg-white text-blue border border-black text-base max-w-80'

    def __init__(self, label: str, td_classes: str = None, label_classes: str = None, tt_text: str = None, tt_classes: str = None):
        super().__init__('td')

        self.classes(self.td_base_classes if td_classes is None else td_classes)
        with self:
            for line in label.split(sep='\n'):
                # add a ' ' in case the line is just '\n'
                ui.label(line + ' ').classes(self.label_base_classes if label_classes is None else label_classes)

            if tt_text is not None:
                # tooltip is on the whole td, not just the label
                # with ui.tooltip().classes(ttclasses if tt_classes is None else tt_classes).props('anchor="bottom middle", self="top middle", offset="[14,14]"') as tt:  # .props('anchor="top middle"'):
                with (ui.tooltip()
                        .classes(self.tt_base_classes if tt_classes is None else tt_classes)
                        .props('anchor="top middle" self="bottom middle" :offset="[14,2]"')) as tt:
                    ui.label(tt_text).classes('max-w-80 font-bold')
