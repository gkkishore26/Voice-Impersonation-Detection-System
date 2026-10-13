import urllib.request

html = urllib.request.urlopen('http://127.0.0.1:5000/', timeout=15).read().decode()
print('favicon in served HTML:', 'rel="icon"' in html)
js = urllib.request.urlopen('http://127.0.0.1:5000/static/js/script.js', timeout=15).read().decode()
print('60s cap live:', '>= 60' in js)
print('old 15s gone:', '>= 15' not in js)
