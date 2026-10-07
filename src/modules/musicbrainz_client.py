import musicbrainzngs
import re
import string
import time
from urllib.error import URLError
from Levenshtein import ratio
from dataclasses import dataclass
from typing import Optional
from Settings import Settings

from modules.console_colors import ULTRASINGER_HEAD, blue_highlighted, red_highlighted


@dataclass
class SongInfo:
    title: str
    artist: str
    year: Optional[str] = None
    genres: Optional[str] = None
    cover_image_data: Optional[bytes] = None
    cover_url: Optional[str] = None


title_filter = [
    "official video",
    "official music video",
    "Offizielles Musikvideo",
]


MAX_RETRIES = 3

# A MusicBrainz recording is only trusted when both its artist and its title
# are at least this similar to what we searched for. Below that, the search
# hit is a different song and must not replace the input metadata.
MIN_MATCH_SIMILARITY = 0.8
# Similarity assigned when one name is the other plus only qualifier words
# (e.g. "Title" vs. "Title Full HD"): a match, but ranked below exact or
# near-exact matches.
CONTAINED_SIMILARITY = 0.85
# Words that only describe a release/upload and never distinguish songs or
# artists. Extra words outside this set (e.g. a second name) prevent a match.
QUALIFIER_WORDS = frozenset(
    "official video music audio lyric lyrics visualizer hd hq 4k full live version edit radio single "
    "album remaster remastered mix remix explicit clean acoustic original extended mono stereo".split()
)
_BRACKETS_RE = re.compile(r"[(\[{][^)\]}]*[)\]}]")
_FEATURING_RE = re.compile(r"\s+(?:feat\.?|ft\.?|featuring)\s+.*$", re.IGNORECASE)


def __clean_string(s: str) -> str:
    return s.translate(str.maketrans('', '', string.punctuation)).lower().strip()


def _normalize_name(s: str, artist: bool) -> str:
    """Drop bracketed additions and, for artists, a featured-guest suffix."""
    s = _BRACKETS_RE.sub(" ", s or "")
    if artist:
        s = _FEATURING_RE.sub("", s)
    return " ".join(__clean_string(s).split())


def _similarity(a: str, b: str, artist: bool = False) -> float:
    """Similarity of two titles (or artist names) in [0, 1].

    Case, punctuation, bracketed additions ("(Official Video)", "(Live 2023)")
    and, for artists, "feat. X" are ignored. If one name is the other plus
    only qualifier words (see QUALIFIER_WORDS, plus numbers), it still counts
    as a match; any other extra word does not - "Nova" is not "Nova Lights".
    """
    a, b = _normalize_name(a, artist), _normalize_name(b, artist)
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    score = ratio(a, b)
    words_a, words_b = a.split(), b.split()
    shorter, longer = (words_a, words_b) if len(words_a) <= len(words_b) else (words_b, words_a)
    extra = list(longer)
    for w in shorter:
        if w not in extra:
            return score
        extra.remove(w)
    if all(w in QUALIFIER_WORDS or w.isdigit() for w in extra):
        score = max(score, CONTAINED_SIMILARITY)
    return score


def __musicbrainz_request(func):
    for i in range(MAX_RETRIES):
        try:
            return func()
        except musicbrainzngs.musicbrainz.NetworkError:
            time.sleep(1)
        except musicbrainzngs.musicbrainz.AuthenticationError:
            # Cover Art Archive returns 401 for some releases
            return None
    return None


def search_musicbrainz(title: str, artist) -> SongInfo:
    # Musicbrainz API documentation
    # https://python-musicbrainzngs.readthedocs.io/en/latest/api/

    musicbrainzngs.set_useragent("UltraSinger", Settings.APP_VERSION, "https://github.com/rakuri255/UltraSinger")

    # remove from search_string "official video"
    # todo: do we need filter?
    origin_title = title
    origin_artist = artist
    for filter in title_filter:
        title = title.lower().replace(filter.lower(), "").strip()
        if artist is not None:
            artist = artist.lower().replace(filter.lower(), "").strip()

    if artist is None:
        recording = __single_line_search(title)
    else:
        recording = __multi_line_search(artist, title)

    if recording is None:
        print(f"{ULTRASINGER_HEAD} {red_highlighted('No match found')}")
        # Keep what we were given: a known artist is better than "Unknown Artist"
        # (the lyrics lookup and the output name depend on it).
        if origin_artist:
            return SongInfo(title=origin_title, artist=origin_artist)
        return SongInfo(title=origin_title, artist="Unknown Artist")

    artist = recording['artist-credit-phrase']
    title = recording['title']
    print(
        f"{ULTRASINGER_HEAD} Found data on Musicbrainz: Artist={blue_highlighted(artist)} Title={blue_highlighted(title)}")

    year = __get_year(recording)
    genres = __get_genres(recording)
    image_data, image_url = __get_image(recording)

    return SongInfo(title=title, artist=artist, year=year, genres=genres, cover_image_data=image_data, cover_url=image_url)


def __single_line_search(search_string):
    search_string = __clean_string(search_string)
    search_string = __filter_words(search_string)

    artists = __musicbrainz_request(lambda: musicbrainzngs.search_artists(search_string, limit=10, artist=search_string))
    recordings = __musicbrainz_request(lambda: musicbrainzngs.search_recordings(search_string, limit=100, artistname=search_string))

    if artists is None or recordings is None:
        return None

    found_artist = None

    for record in recordings['recording-list']:
        if found_artist is not None:
            break

        for artist_credit in record['artist-credit']:
            if found_artist is not None:
                break
            if isinstance(artist_credit, str):
                continue
            # todo: there is also an "alias-list". Maybe search also there?

            for artist in artists['artist-list']:
                if artist_credit['artist'] and artist_credit['artist']['id'] == artist['id']:
                    found_artist = record['artist-credit-phrase']
                    break

    if found_artist is None:
        return None

    recordings = [x for x in recordings['recording-list'] if
                  __clean_string(x['artist-credit-phrase']) == __clean_string(found_artist)]

    recording = None

    for record in recordings:
        if __clean_string(record['title']) in __clean_string(search_string):
            recording = record

    return recording


def __filter_words(search_string):
    for filter in title_filter:
        search_string = search_string.lower().replace(filter.lower(), "").strip()
    return search_string


def __multi_line_search(artist: str, title: str):
    # Try both combinations since we don't know which one is the artist and which one is the title
    artist1, title1 = artist, title
    artist2, title2 = title, artist

    result1 = __musicbrainz_request(lambda: musicbrainzngs.search_recordings(recording=title1, limit=10, artist=artist1, artistname=artist1))
    result2 = __musicbrainz_request(lambda: musicbrainzngs.search_recordings(recording=title2, limit=10, artist=artist2, artistname=artist2))

    if result1 is None:
        result1 = {'recording-count': 0, 'recording-list': []}
    if result2 is None:
        result2 = {'recording-count': 0, 'recording-list': []}

    # Only accept a recording whose artist AND title match what we searched
    # for; the best-matching one wins. Previously the first search hit was
    # taken even when neither matched, replacing a correct "Artist - Title"
    # with an unrelated song (and the lyrics lookup then fetched that song).
    best, best_score = None, 0.0
    for result, wanted_artist, wanted_title in ((result1, artist1, title1), (result2, artist2, title2)):
        for record in result.get('recording-list', []):
            artist_sim = _similarity(record.get('artist-credit-phrase', ''), wanted_artist, artist=True)
            title_sim = _similarity(record.get('title', ''), wanted_title)
            if artist_sim < MIN_MATCH_SIMILARITY or title_sim < MIN_MATCH_SIMILARITY:
                continue
            if artist_sim + title_sim > best_score:
                best, best_score = record, artist_sim + title_sim
    return best


def __get_image(recording) -> (bytes, str):
    image_data = None
    image_url = None
    if 'release-list' in recording:
        for release in recording['release-list']:
            try:
                image_data = __musicbrainz_request(lambda: musicbrainzngs.get_image_front(release['id']))
                if image_data is None:
                    continue

                image_list = __musicbrainz_request(lambda: musicbrainzngs.get_image_list(release['id']))
                if image_list is None:
                    continue

                for image in image_list['images']:
                    if image['front']:
                        image_url = image['image']
                        break
                break
            except Exception as e:
                # Catch all exceptions (including CAA errors) to prevent
                # cover art failures from crashing the entire pipeline.
                # A 404 just means THIS release has no cover in the Cover
                # Art Archive — an expected, routine case while iterating
                # the recording's releases, so don't alarm the user; the
                # loop simply tries the next release. Detect it from the
                # real HTTP status: musicbrainzngs wraps CAA failures in
                # ResponseError(cause=HTTPError) exposing .cause.code, a
                # bare HTTPError carries .code directly.
                status = getattr(e, "code", None)
                if status is None:
                    status = getattr(getattr(e, "cause", None), "code", None)
                if status != 404:
                    print(f"{ULTRASINGER_HEAD} Cover art download failed: {e}")
                continue
    if image_data is not None:
        print(f"{ULTRASINGER_HEAD} Found cover image")
    elif 'release-list' in recording:
        # Informational only: for URL input the video thumbnail is used as
        # the cover instead; local files simply get no MusicBrainz cover.
        print(
            f"{ULTRASINGER_HEAD} No cover art on MusicBrainz for this recording."
        )

    return image_data, image_url


def __get_year(recording):
    year = None

    # Recordings without any release (empty list) have no year either
    if not recording.get('release-list'):
        return year

    release_group_id = recording['release-list'][0]['release-group']['id']
    release_group = __musicbrainz_request(lambda: musicbrainzngs.get_release_group_by_id(release_group_id))

    if release_group is None:
        return year

    if 'first-release-date' not in release_group['release-group']:
        return year

    year = release_group['release-group']['first-release-date'].strip()
    year = year.split('-')[0]

    if year is not None:
        print(f"{ULTRASINGER_HEAD} Found year: {blue_highlighted(year)}")

    return year


def __get_genres(recording) -> str:
    # todo secondary-type-list ??
    genres = None
    if 'tag-list' in recording:
        genres = ""
        for tag in recording['tag-list']:
            genres += f"{tag['name'].strip()},"
    if genres is not None:
        print(f"{ULTRASINGER_HEAD} Found genres: {blue_highlighted(genres)}")
    return genres
