import musicbrainzngs
import re
import socket
import string
import threading
import time
from contextlib import contextmanager
from urllib.error import URLError
from dataclasses import dataclass
from typing import Optional
from Settings import Settings

from modules.console_colors import ULTRASINGER_HEAD, blue_highlighted, gold_highlighted, red_highlighted
from modules.name_matching import (
    MIN_MATCH_SIMILARITY,
    SAME_RECORDING_S,
    bracketed_parts,
    similarity,
    strip_title_additions,
    title_before_dash,
    title_key,
    titles_match,
    without_brackets,
)


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
# Seconds a connection to MusicBrainz or the Cover Art Archive may stay silent.
# musicbrainzngs sets no timeout: a server that never answered blocked the
# conversion for good.
REQUEST_TIMEOUT_SECONDS = 10
# A lookup (search, year and cover) makes no more requests once its failed ones
# took this long. musicbrainzngs already retries a request up to 8 times, for
# about a minute when the server is overloaded; retrying that again, for each of
# up to MAX_COVER_RELEASES releases, took 43 minutes before "No cover art".
FAILED_REQUESTS_SECONDS = 30
# Recordings asked for per artist+title search. A well-known song has dozens
# of recordings with the same title (album, single, live, demos, videos) that
# the search ranks alike, and the year and cover come from all that match.
RECORDINGS_PER_SEARCH = 100
# At most this many releases are asked for a cover, the preferred ones first
MAX_COVER_RELEASES = 15
# A part or volume number ("Part 2", "Vol. 3"), not a year
_NUMBER_RE = re.compile(r"(?<!\d)\d{1,3}(?!\d)")


def __clean_string(s: str) -> str:
    return s.translate(str.maketrans('', '', string.punctuation)).lower().strip()


# Lookups can run in several threads at a time (the GUI's lyrics badge looks
# songs up in the background). Per thread: failed_seconds, the seconds that
# the failed requests of its current lookup took.
_lookup = threading.local()
# The requests running with the socket timeout, and the default timeout
# before the first of them
_timeout_lock = threading.Lock()
_timeout_users = 0
_timeout_before = None


@contextmanager
def _socket_timeout(seconds: float):
    """Connections opened meanwhile time out after ``seconds`` of silence.

    musicbrainzngs opens its connections without a timeout; the default
    timeout of new sockets, set only while its request runs, gives them one.
    Requests of several threads share it: the default timeout before the
    first one comes back when the last one has ended.
    """
    global _timeout_users, _timeout_before
    with _timeout_lock:
        if not _timeout_users:
            _timeout_before = socket.getdefaulttimeout()
        _timeout_users += 1
        socket.setdefaulttimeout(seconds)
    try:
        yield
    finally:
        with _timeout_lock:
            _timeout_users -= 1
            if not _timeout_users:
                socket.setdefaulttimeout(_timeout_before)


def _http_status(error) -> Optional[int]:
    """HTTP status of a failed request: musicbrainzngs wraps the HTTPError in
    ResponseError(cause=HTTPError), a bare HTTPError carries it itself."""
    status = getattr(error, "code", None)
    if status is None:
        status = getattr(getattr(error, "cause", None), "code", None)
    return status


def __musicbrainz_request(func, verbose: bool = True):
    """The answer of a request to the song database, or None when there is
    none: nothing found, or the database does not answer (see
    FAILED_REQUESTS_SECONDS). ``verbose`` reports a failed request."""
    for attempt in range(MAX_RETRIES):
        failed_seconds = getattr(_lookup, "failed_seconds", 0.0)
        if failed_seconds >= FAILED_REQUESTS_SECONDS:
            return None
        started = time.monotonic()
        if attempt:
            time.sleep(1)
        try:
            with _socket_timeout(REQUEST_TIMEOUT_SECONDS):
                return func()
        except musicbrainzngs.musicbrainz.NetworkError as e:
            # Unreachable, overloaded or silent
            _lookup.failed_seconds = failed_seconds + time.monotonic() - started
            if _lookup.failed_seconds >= FAILED_REQUESTS_SECONDS and verbose:
                print(f"{ULTRASINGER_HEAD} {gold_highlighted('Warning:')} MusicBrainz does not answer ({e}) "
                      f"- going on without it")
        except musicbrainzngs.musicbrainz.AuthenticationError:
            # Cover Art Archive returns 401 for some releases
            return None
        except musicbrainzngs.musicbrainz.ResponseError as e:
            # Not found or a bad request, e.g. a release group that was merged
            # into another, or a release without a cover: no answer, as when
            # nothing matches
            if _http_status(e) != 404 and verbose:
                print(f"{ULTRASINGER_HEAD} MusicBrainz request failed: {e}")
            return None
    return None


def search_musicbrainz(title: str, artist, quick: bool = False, duration: float | None = None) -> SongInfo:
    """The song in the song database: its artist, title, year, genres and cover.

    When nothing matches a title with a description after a dash ("Title -
    Singer Name live in the studio"), the title before the dash is searched
    (see ``title_before_dash``). Last, with a known ``duration`` (the length
    of the audio, in seconds), the title before a dash whatever follows it
    ("Title - Chapter 2") is searched, taking only recordings of about the
    length of the audio (see ``SAME_RECORDING_S``). ``quick`` only looks up
    the artist and the title, and reports nothing (for the GUI's lyrics
    badge, which names the song as a conversion does).
    """
    # Musicbrainz API documentation
    # https://python-musicbrainzngs.readthedocs.io/en/latest/api/

    _lookup.failed_seconds = 0.0
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
        recordings = [r for r in [__single_line_search(title, not quick)] if r is not None]
    else:
        recordings = __multi_line_search(artist, title, not quick)
        searched = {_search_key(title)}
        # "Title - Singer Name live in the studio": the song is "Title"
        shorter = title_before_dash(origin_title)
        if not recordings and shorter:
            if not quick:
                print(f"{ULTRASINGER_HEAD} Searching again with the title before the dash: "
                      f"{blue_highlighted(shorter)}")
            recordings = __multi_line_search(artist, shorter, not quick)
            searched.add(_search_key(shorter))
        # "Title - Chapter 2": "Title - Part 2" can be another song than
        # "Title", so only a recording of the length of the audio shows that
        # the song is "Title"
        before_dash = title_before_dash(origin_title, any_rest=True) if duration else ""
        if not recordings and before_dash and _search_key(before_dash) not in searched:
            if not quick:
                print(f"{ULTRASINGER_HEAD} Searching again with the title before the dash, for a recording "
                      f"of the length of the audio: {blue_highlighted(before_dash)}")
            recordings = __multi_line_search(artist, before_dash, not quick, length=duration)

    if not recordings:
        if not quick:
            print(f"{ULTRASINGER_HEAD} {red_highlighted('No match found')}")
        # Keep what we were given: a known artist is better than "Unknown Artist"
        # (the lyrics lookup and the output name depend on it).
        if origin_artist:
            return SongInfo(title=origin_title, artist=origin_artist)
        return SongInfo(title=origin_title, artist="Unknown Artist")

    recording = recordings[0]
    artist = recording['artist-credit-phrase']
    title = recording['title']
    if quick:
        return SongInfo(title=title, artist=artist)
    print(
        f"{ULTRASINGER_HEAD} Found data on Musicbrainz: Artist={blue_highlighted(artist)} Title={blue_highlighted(title)}")

    # The year and the cover come from the releases of all recordings of the
    # song: which one the search lists first (a live one, a video) is chance,
    # and its releases can all be later ones.
    releases = _releases_of(_same_song(recordings))
    year = __get_year(releases)
    genres = __get_genres(recording)
    image_data, image_url = __get_image(releases)

    return SongInfo(title=title, artist=artist, year=year, genres=genres, cover_image_data=image_data, cover_url=image_url)


def _releases_of(recordings) -> list:
    """The releases of the recordings, each once."""
    releases, seen = [], set()
    for recording in recordings:
        for release in recording.get('release-list') or []:
            key = release.get('id') or id(release)
            if key not in seen:
                seen.add(key)
                releases.append(release)
    return releases


def __single_line_search(search_string, verbose: bool = True):
    search_string = __clean_string(search_string)
    search_string = __filter_words(search_string)

    artists = __musicbrainz_request(
        lambda: musicbrainzngs.search_artists(search_string, limit=10, artist=search_string), verbose)
    recordings = __musicbrainz_request(
        lambda: musicbrainzngs.search_recordings(search_string, limit=100, artistname=search_string), verbose)

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


def _search_key(title: str) -> str:
    """The title that a search asks for (see __multi_line_search), to tell
    titles already searched: case and spaces aside."""
    return " ".join(strip_title_additions(title).split()).casefold()


def __multi_line_search(artist: str, title: str, verbose: bool = True, length: float | None = None) -> list:
    """The recordings that match the artist and the title, the best first.

    The title is searched without its additions (see strip_title_additions):
    with "(US Version)" or "- Live" in the search, a version of the song was
    found first, or no recording at all. With ``length`` (in seconds), only
    recordings of that length count (see _has_length).
    """
    title = strip_title_additions(title)
    # Try both combinations since we don't know which one is the artist and which one is the title
    artist1, title1 = artist, title
    artist2, title2 = title, strip_title_additions(artist)

    # "status" only ranks recordings on official releases higher (the search
    # fields are alternatives): a famous song has hundreds of equally ranked
    # recordings, mostly live bootlegs, and without it the hundred listed can
    # miss the original recording entirely
    result1 = __musicbrainz_request(lambda: musicbrainzngs.search_recordings(
        recording=title1, limit=RECORDINGS_PER_SEARCH, artist=artist1, artistname=artist1, status="official"), verbose)
    result2 = __musicbrainz_request(lambda: musicbrainzngs.search_recordings(
        recording=title2, limit=RECORDINGS_PER_SEARCH, artist=artist2, artistname=artist2, status="official"), verbose)

    if result1 is None:
        result1 = {'recording-count': 0, 'recording-list': []}
    if result2 is None:
        result2 = {'recording-count': 0, 'recording-list': []}

    searches = ((result1, artist1, title1), (result2, artist2, title2))
    # A bracketed part of the wanted title that is no known addition can
    # name another song ("(Part 2)"), so it has to match first; only when no
    # recording does, the title is compared without it, as all titles were
    # before: video titles carry more additions than any list knows. Not
    # when such a part holds a part or volume number (a year is fine).
    recordings = _matching_recordings(searches, strict=True, length=length)
    if recordings or any(_NUMBER_RE.search(part) for part in bracketed_parts(title1) + bracketed_parts(title2)):
        return recordings
    return _matching_recordings(searches, strict=False, length=length)


def _matching_recordings(searches, strict: bool, length: float | None = None) -> list:
    """The recordings of the search results whose artist AND title match what
    we searched for, the best-matching first.

    Previously the first search hit was taken even when neither matched,
    replacing a correct "Artist - Title" with an unrelated song (and the
    lyrics lookup then fetched that song). Of equally good ones, a title
    without additions comes first: "Title" rather than "Title (Live)".
    The titles are compared with ``titles_match``; unless ``strict``, the
    wanted title without its bracketed parts (the recording's title keeps
    them: "Title (Part 2)" still is another song than "Title"). With
    ``length`` (in seconds), only recordings of that length count.
    """
    matches = []
    for result, wanted_artist, wanted_title in searches:
        compared_title = wanted_title if strict else without_brackets(wanted_title)
        for record in result.get('recording-list', []):
            record_title = record.get('title', '')
            artist_sim = similarity(record.get('artist-credit-phrase', ''), wanted_artist, artist=True)
            if artist_sim < MIN_MATCH_SIMILARITY or not titles_match(record_title, compared_title):
                continue
            if length is not None and not _has_length(record, length):
                continue
            title_sim = similarity(record_title, wanted_title)
            plain = strip_title_additions(record_title) == record_title.strip()
            matches.append(((artist_sim + title_sim, plain), record))
    # Stable: equally good recordings keep the order of the search
    matches.sort(key=lambda match: match[0], reverse=True)
    recordings, seen = [], set()
    for _, record in matches:
        key = record.get('id') or id(record)
        if key not in seen:
            seen.add(key)
            recordings.append(record)
    return recordings


def _has_length(recording, length: float) -> bool:
    """Whether the recording is at most SAME_RECORDING_S off ``length``
    seconds long. The song database gives the length in milliseconds, when
    it knows it; a recording without a length is not of that length."""
    try:
        seconds = float(recording.get('length')) / 1000
    except (TypeError, ValueError):
        return False
    return seconds > 0 and abs(seconds - length) <= SAME_RECORDING_S


def _same_song(recordings) -> list:
    """The recordings with the title of the first one, its additions aside
    ("Title" and "Title (Live)", not "Titles" or "Title (Part 2)"): only
    those date the song."""
    first = title_key(recordings[0].get('title', ''))
    return [r for r in recordings if title_key(r.get('title', '')) == first]


def __get_image(releases) -> (bytes, str):
    image_data = None
    image_url = None
    if releases:
        # The cover of the earliest official release (the original single or
        # album), not of the first listed one, which is often a compilation or
        # a bootleg
        for release in _preferred_releases(releases)[:MAX_COVER_RELEASES]:
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
                # loop simply tries the next release (__musicbrainz_request
                # already answers it with None).
                if _http_status(e) != 404:
                    print(f"{ULTRASINGER_HEAD} Cover art download failed: {e}")
                continue
    if image_data is not None:
        print(f"{ULTRASINGER_HEAD} Found cover image")
    elif releases:
        # Informational only: for URL input the video thumbnail is used as
        # the cover instead; local files simply get no MusicBrainz cover.
        print(
            f"{ULTRASINGER_HEAD} No cover art on MusicBrainz for this song."
        )

    return image_data, image_url


def _year_of(date) -> Optional[str]:
    """The year of a MusicBrainz date ("1989", "1989-03" or "1989-03-21"), or None."""
    year = (date or "").strip()[:4]
    return year if len(year) == 4 and year.isdigit() else None


def _is_official(release) -> bool:
    return (release.get('status') or '').strip().lower() == 'official'


# Kinds of release groups where a song rarely came out first; their releases
# come after the albums, singles and EPs (e.g. for the cover)
_LATER_KINDS = frozenset({"compilation", "live", "remix", "dj-mix", "mixtape/street", "demo"})


def _is_later_kind(release) -> bool:
    group = release.get('release-group') or {}
    kinds = [group.get('type') or ''] + list(group.get('secondary-type-list') or [])
    return any(kind.strip().lower() in _LATER_KINDS for kind in kinds)


def _preferred_releases(releases):
    """The releases of a song: official ones first, of those the albums,
    singles and EPs before compilations, live and remix releases, each from
    the earliest date on (undated ones last). MusicBrainz does not list them
    by date."""
    return sorted(releases, key=lambda r: (not _is_official(r), _is_later_kind(r), not _year_of(r.get('date')),
                                           (r.get('date') or '').strip()))


def __get_year(releases):
    """Year of the earliest official album, single or EP among the song's
    releases (of the earliest official release, when there is none of these;
    of the earliest release, when none is official).

    The releases are not listed by date, and the first one is often a later
    compilation. The release group of the earliest release can know a still
    earlier date (a reissue's group starts with the original release).
    """
    # Recordings without any release (empty list) have no year either
    if not releases:
        return None

    ordered = _preferred_releases(releases)
    # Bootlegs and promotional copies do not date a song that has an official
    # release, even an undated one (its release group may know the date);
    # nor do compilations and live releases one that has an official album,
    # single or EP. A group of releases that knows no date at all gives way
    # to the next one.
    official = [r for r in ordered if _is_official(r)]
    tried = []
    for pool in ([r for r in official if not _is_later_kind(r)], official, ordered):
        if not pool or pool in tried:
            continue
        tried.append(pool)
        year = _earliest_year(pool)
        if year:
            print(f"{ULTRASINGER_HEAD} Found year: {blue_highlighted(year)}")
            return year
    return None


def _earliest_year(pool) -> Optional[str]:
    """The earliest year of the dated releases in ``pool`` (in preferred
    order) and of the release group of the first of them."""
    dated = [r for r in pool if _year_of(r.get('date'))]
    years = [_year_of(r['date']) for r in dated]
    earliest = dated[0] if dated else pool[0]

    release_group_id = (earliest.get('release-group') or {}).get('id')
    if release_group_id:
        release_group = __musicbrainz_request(lambda: musicbrainzngs.get_release_group_by_id(release_group_id))
        if release_group is not None:
            group_year = _year_of(release_group.get('release-group', {}).get('first-release-date'))
            if group_year:
                years.append(group_year)
    return min(years) if years else None


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
