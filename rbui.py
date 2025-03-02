from nicegui import ui


#                           'table {width:100%;table-layout:fixed;} '
#                           'td {border:1px solid;padding: 8px;} '
#                           'dt {padding: 0;line-height: 1.0;} '
#                           'dd {padding: 0;line-height: 1.5;margin-left: 30px} '
#                           'th {border: 1px solid; padding: 2px;} '

# noinspection PyPep8Naming
class table(ui.element):
    default_table_classes = 'border-solid border border-white w-11/12 table-fixed'

    def __init__(self, table_classes: str = None):
        super().__init__('table')
        self.classes(self.default_table_classes if table_classes is None else table_classes)


# noinspection PyPep8Naming
class tr(ui.element):
    def __init__(self, tr_classes: str = None):
        super().__init__('tr')
        trclasses = ''
        self.classes(trclasses if tr_classes is None else tr_classes)


# noinspection PyPep8Naming
class th(ui.element):
    th_base_classes = 'font-bold border-solid border border-white p-2'
    label_base_classes = 'text-base'

    def __init__(self, label: str, th_classes: str = None, label_classes: str = None, th_props: str = None):
        super().__init__(f'th')
        if th_props is not None:  # for html attributes
            self.props(th_props)
        self.classes(self.th_base_classes if th_classes is None else th_classes)
        with self:
            ui.label(label).classes(self.label_base_classes if label_classes is None else label_classes)


# noinspection PyPep8Naming
class td(ui.element):
    td_base_classes = 'border-solid border border-white p-2'
    label_base_classes = 'text-base'
    tt_base_classes = 'bg-white text-blue border border-white text-base max-w-80'

    def __init__(self, label: str, td_classes: str = None, label_classes: str = None, td_props: str = None, td_style: str = None, tt_text: str = None, tt_classes: str = None):
        super().__init__('td')
        if td_props is not None:
            self.props(td_props)

        self.classes(self.td_base_classes if td_classes is None else td_classes)
        if td_style is not None:
            self.style(td_style)
        with self:
            with ui.column().classes('gap-y-0'):
                for line in label.split(sep='\n'):
                    # add a ' ' in case the line is just '\n'
                    ui.label(line + ' ').classes(self.label_base_classes if label_classes is None else label_classes)

                if tt_text is not None:
                    # tooltip is on the whole td, not just the label
                    # with ui.tooltip().classes(ttclasses if tt_classes is None else tt_classes).props('anchor="bottom middle", self="top middle", offset="[14,14]"') as tt:  # .props('anchor="top middle"'):
                    with (ui.tooltip()
                                  .classes(self.tt_base_classes if tt_classes is None else tt_classes)
                                  .props('anchor="top middle" self="bottom middle" :offset="[14,2]"')):
                        ui.label(tt_text).classes('max-w-80 font-bold')
