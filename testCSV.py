import csv
with open('eggs.csv', 'wb') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=' ',quoting=csv.QUOTE_MINIMAL)
    spamwriter.writerow(['Spam', 'Lovely Spam', 'Wonderful Spam'])