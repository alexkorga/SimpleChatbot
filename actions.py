import datetime as dt


def chat_quit():
    print('Auf Wiedersehen!')


def get_time():
    time = dt.datetime.now().time()
    current_time = time.strftime('%H:%M')
    print(f'Es ist grade {current_time} Uhr.')


def get_date():
    day_number = dt.datetime.today().weekday()
    week_days = ('Montag', 'Dienstag', 'Mittwoch', 'Donnerstag', 'Freitag', 'Samstag', 'Sonntag')

    date = dt.date.today()
    current_date = date.strftime('%d.%m.%Y')
    print(f'Heute ist {week_days[day_number]}, der {current_date}.')


def action_manager(action):
    if 'Time' in action:
        get_time()
        return False
    elif 'Date' in action:
        get_date()
        return False
    elif 'Quit' in action:
        chat_quit()
        return True
