from pelican.plugins import render_math

PATH = 'content'
ARTICLE_PATHS = ['Articles']
ARTICLE_SAVE_AS = '{date:%Y}/{slug}.html'
ARTICLE_URL = '{date:%Y}/{slug}.html'

SUMMARY_MAX_LENGTH = 100


AUTHOR = 'HsiaoEn'
SITENAME = "HsiaoEn's Blog"
SITEURL = ''

TIMEZONE = 'Asia/Taipei'

DEFAULT_LANG = 'en'
DEFAULT_DATE_FORMAT = '%Y-%m-%d'

# Theme
THEME = './pelican-themes/foundation-default-colours' 
FOUNDATION_FOOTER_TEXT = '''© 2022 Sun, Hsiao-En <a href = "feeds/all.atom.xml"><img src= "../images/logo_rss.png" width = "28"></a>
                            <a href = "mailto:hsiaoen.sum@gmail.com"><img src= "../images/logo_email.png" width = "28"></a>
                            <a href = "https://github.com/HsiaoEn"><img src= "../images/logo_github.png" width = "28"></a> 
                            <a href = "https://github.com/HsiaoEn"><img src= "../images/logo_linkedIn.png" width = "28"></a> 
                            '''

# Feed generation is usually not desired when developing
FEED_ALL_ATOM = 'feeds/all.atom.xml'
FEED_ALL_RSS = 'feeds/all.rss.xml'
AUTHOR_FEED_RSS = 'feeds/%s.rss.xml'
RSS_FEED_SUMMARY_ONLY = False

# Blogroll
LINKS = (('GitHub', 'https://github.com/HsiaoEn'),
         ('LinkedIn', 'https://www.linkedin.com/in/hsiao-en-sun-538644180/'),
         )

# Social widget
SOCIAL = (('You can add links in your config file', '#'),
          ('Another social link', '#'),)

DEFAULT_PAGINATION = False

# Pages
DISPLAY_PAGES_ON_MENU = True

# Uncomment following line if you want document-relative URLs when developing
#RELATIVE_URLS = True


