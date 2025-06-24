def calculate_average_performance(domain, p1, p2, p3):
    if domain == 'NLP':
        return round((p1 + p2 + p3) / 3, 3)
    elif domain == 'SE':
        return round((p1*84 + p2*157 + p3*167)/408, 3)


if __name__ == '__main__':
    print(calculate_average_performance('SE', 0.274,
0.166,
0.551,


))
