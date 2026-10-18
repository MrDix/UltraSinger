"""Test the musicbrainz_client module."""

import io
import socket
import threading
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch
from urllib.error import HTTPError

import musicbrainzngs

from src.modules import musicbrainz_client
from src.modules.musicbrainz_client import search_musicbrainz


class TestGetMusicInfos(unittest.TestCase):

    @patch('musicbrainzngs.search_artists')
    @patch('musicbrainzngs.search_recordings')
    @patch('musicbrainzngs.get_image_front')
    @patch('musicbrainzngs.get_image_list')
    @patch('musicbrainzngs.get_release_group_by_id')
    def test_get_music_infos(self, mock_get_release_group_by_id, mock_get_image_list, mock_get_image_front, mock_search_recordings, mock_search_artists):
        # Arrange
        artist = 'UltraSinger'
        title = 'That\'s Rocking! (UltrStar 2023) FULL HD'

        # Set up mock return values for the MusicBrainz API calls
        mock_search_artists.return_value = {
            'artist-list': [
                {
                    'id': 'fake_artist_id',
                    'name': artist
                }
            ]
        }

        # image_data = musicbrainzngs.get_image_front(release['id'])
        mock_get_image_front.return_value = b'fake image data'


        mock_get_image_list.return_value = {
            'images': [
                {
                    'front': True,
                    'image': 'https://example.com/image.jpg'
                }
            ]
        }

        mock_get_release_group_by_id.return_value = {
            'release-group': {
                'first-release-date': '2023-01-01'
            }
        }

        mock_search_recordings.return_value = {
            'recording-list': [
                {
                    'title': 'That\'s Rocking!',
                    'artist-credit-phrase': artist,
                    'release-list': [
                        {
                            'id': 'fake_release_id',
                            'release-group': {
                                'id': 'fake_group_id',
                                }
                        },
                    ],
                    'tag-list': [
                        {'name': 'Genre 1'},
                        {'name': 'Genre 2'},
                    ],
                    'artist-credit': [
                        {
                            'artist': {'id': 'fake_artist_id'}
                        }
                    ],
                }
            ]}

        # Call the function to test
        song_info_single_line = search_musicbrainz(f'{artist} - {title}', None) # Single line test

        # Assert the returned values
        self.assertEqual(song_info_single_line.title, 'That\'s Rocking!')
        self.assertEqual(song_info_single_line.artist, 'UltraSinger')
        self.assertEqual(song_info_single_line.year, '2023')
        self.assertEqual(song_info_single_line.genres, 'Genre 1,Genre 2,')
        self.assertEqual(song_info_single_line.cover_image_data, b'fake image data')
        self.assertEqual(song_info_single_line.cover_url, 'https://example.com/image.jpg')

        song_info_multi_line = search_musicbrainz(title, artist) # multi line test

        self.assertEqual(song_info_multi_line.title, 'That\'s Rocking!')
        self.assertEqual(song_info_multi_line.artist, 'UltraSinger')
        self.assertEqual(song_info_multi_line.year, '2023')
        self.assertEqual(song_info_multi_line.genres, 'Genre 1,Genre 2,')
        self.assertEqual(song_info_multi_line.cover_image_data, b'fake image data')
        self.assertEqual(song_info_multi_line.cover_url, 'https://example.com/image.jpg')




    @patch('musicbrainzngs.search_artists')
    @patch('musicbrainzngs.search_recordings')
    def test_get_empty_artist_music_infos(self, mock_search_recordings, mock_search_artists):
        # Arrange
        artist = 'UltraSinger'
        title = 'That\'s Rocking! (UltrStar 2023) FULL HD'

        # Set up mock return values for the MusicBrainz API calls
        mock_search_artists.return_value = {
            'artist-list': []
        }

        mock_search_recordings.return_value = {
            'recording-list': [
                {
                    'title': 'That\'s Rocking!',
                    'artist-credit-phrase': artist,
                    'release-list': [
                        {
                            'id': 'fake_release_id',
                            'release-group': {
                                'id': 'fake_group_id',
                                }
                        },
                    ],
                    'tag-list': [
                        {'name': 'Genre 1'},
                        {'name': 'Genre 2'},
                    ],
                    'artist-credit': [
                        {
                            'artist': {'id': 'fake_artist_id'}
                        }
                    ],
                }
            ]}

        # Act
        song_info_single_line = search_musicbrainz(f'{artist} - {title}', None) # Single line test

        # Assert
        self.assertEqual(song_info_single_line.title, f'{artist} - {title}')
        self.assertEqual(song_info_single_line.artist, "Unknown Artist")
        self.assertEqual(song_info_single_line.year, None)
        self.assertEqual(song_info_single_line.genres, None)
        self.assertEqual(song_info_single_line.cover_image_data, None)
        self.assertEqual(song_info_single_line.cover_url, None)

    @patch('musicbrainzngs.search_artists')
    @patch('musicbrainzngs.search_recordings')
    def test_get_empty_release_music_infos(self, mock_search_recordings, mock_search_artists):
        # Arrange
        artist = 'UltraSinger'
        title = 'That\'s Rocking! (UltrStar 2023) FULL HD'

        # Set up mock return values for the MusicBrainz API calls
        mock_search_artists.return_value = {
            'artist-list': [
                {'name': f'  {artist}  '}  # Also test leading and trailing whitespaces
            ]
        }

        mock_search_recordings.return_value = {
            'recording-list': []
        }

        # Act
        song_info_single_line = search_musicbrainz(f'{artist} - {title}', None) # Single line test

        # Assert
        self.assertEqual(song_info_single_line.title, f'{artist} - {title}')
        self.assertEqual(song_info_single_line.artist, "Unknown Artist")
        self.assertEqual(song_info_single_line.year, None)
        self.assertEqual(song_info_single_line.genres, None)
        self.assertEqual(song_info_single_line.cover_image_data, None)
        self.assertEqual(song_info_single_line.cover_url, None)


    @unittest.skip("Search with real data only test manually")
    def test_search_musicbrainz_with_real_data(self):

        # Arrange
        search_list = [
            # (search_artist, search_title, expected_artist, expected_title)
            # Use your own test data here — do not commit copyrighted song/artist names.
            (None, 't', None, None),  # this should return "Unknown artist"
        ]

        failed = 0
        success = 0
        count = 0
        for i, search_string in enumerate(search_list):
            artist = search_string[0]
            title = search_string[1]
            print(f"({i}) - {artist} - {title}")
            count = i

            song_info = search_musicbrainz(title, artist)
            print(f'\t{search_string}\t -> {song_info.artist}, {song_info.title}, {song_info.year}, {song_info.genres}')
            print('-------------------------------')
        print(f"Failed: {failed} | Success: {success} Count: {count}")


if __name__ == '__main__':
    unittest.main()


class TestCoverArt404Detection(unittest.TestCase):
    """Routine 404s are detected via the real HTTP status, not str(e)."""

    @staticmethod
    def _get_image_fn():
        from src.modules import musicbrainz_client as mb
        # Module-level dunder name: fetch via __dict__ to dodge the test
        # class's own name mangling.
        return mb, mb.__dict__["__get_image"]

    def _run(self, http_status):
        import io as _io
        from contextlib import redirect_stdout
        from unittest.mock import patch
        from urllib.error import HTTPError
        import musicbrainzngs

        mb, get_image = self._get_image_fn()
        cause = HTTPError("u", http_status, "msg", None, None)
        exc = musicbrainzngs.ResponseError(cause=cause)
        releases = [{"id": "r1"}]
        buf = _io.StringIO()
        with patch.object(mb.musicbrainzngs, "get_image_front",
                          side_effect=exc), redirect_stdout(buf):
            result = get_image(releases)
        return result, buf.getvalue()

    def test_404_cause_is_silent(self):
        result, out = self._run(404)
        self.assertEqual(result, (None, None))
        self.assertNotIn("Cover art download failed", out)
        self.assertIn("No cover art on MusicBrainz", out)

    def test_other_status_is_loud(self):
        _, out = self._run(400)
        self.assertIn("MusicBrainz request failed", out)


def _recording(title, artist):
    return {'title': title, 'artist-credit-phrase': artist, 'release-list': [],
            'artist-credit': [{'artist': {'id': 'x'}}]}


class TestMultiLineMatchValidation(unittest.TestCase):
    """Artist+title searches only accept recordings that really match."""

    @patch('musicbrainzngs.search_recordings')
    def test_unrelated_hit_keeps_input_metadata(self, mock_search):
        # The search returns a different artist and song only: it must not
        # replace the artist/title we searched for.
        mock_search.return_value = {'recording-count': 1,
                                    'recording-list': [_recording('Other Song', 'Other Band')]}
        info = search_musicbrainz('Wanted Title', 'Wanted Artist')
        self.assertEqual(info.artist, 'Wanted Artist')
        self.assertEqual(info.title, 'Wanted Title')

    @patch('musicbrainzngs.search_recordings')
    def test_no_hit_keeps_input_artist(self, mock_search):
        mock_search.return_value = {'recording-count': 0, 'recording-list': []}
        info = search_musicbrainz('Wanted Title', 'Wanted Artist')
        self.assertEqual((info.artist, info.title), ('Wanted Artist', 'Wanted Title'))

    @patch('musicbrainzngs.search_recordings')
    def test_best_title_wins_not_first(self, mock_search):
        mock_search.return_value = {'recording-count': 2, 'recording-list': [
            _recording('Different Song', 'Wanted Artist'),
            _recording('Wanted Title', 'Wanted Artist'),
        ]}
        info = search_musicbrainz('Wanted Title', 'Wanted Artist')
        self.assertEqual(info.title, 'Wanted Title')

    @patch('musicbrainzngs.search_recordings')
    def test_same_artist_wrong_song_rejected(self, mock_search):
        mock_search.return_value = {'recording-count': 1,
                                    'recording-list': [_recording('Completely Else', 'Wanted Artist')]}
        info = search_musicbrainz('Wanted Title', 'Wanted Artist')
        self.assertEqual(info.title, 'Wanted Title')

    @patch('musicbrainzngs.search_recordings')
    def test_spelling_variants_accepted(self, mock_search):
        mock_search.return_value = {'recording-count': 1, 'recording-list': [
            _recording("Wanted Title", 'Wanted Artist feat. Guest')]}
        info = search_musicbrainz('Wanted Title (Official Video)', 'WANTED ARTIST')
        self.assertEqual((info.artist, info.title), ('Wanted Artist feat. Guest', 'Wanted Title'))

    @patch('musicbrainzngs.search_recordings')
    def test_swapped_artist_and_title(self, mock_search):
        # file named "Title - Artist": the second ordering must find it
        def search(recording=None, limit=None, artist=None, artistname=None, **fields):
            if artist == 'wanted artist':
                return {'recording-count': 1, 'recording-list': [_recording('Wanted Title', 'Wanted Artist')]}
            return {'recording-count': 0, 'recording-list': []}
        mock_search.side_effect = search
        info = search_musicbrainz('Wanted Artist', 'Wanted Title')
        self.assertEqual((info.artist, info.title), ('Wanted Artist', 'Wanted Title'))


class TestYear(unittest.TestCase):
    """The year comes from the earliest release of the recording, not from its first listed one."""

    @staticmethod
    def _recording(*releases):
        rec = _recording('Wanted Title', 'Wanted Artist')
        rec['release-list'] = [{'id': f'r{i}', 'title': t, 'date': d, 'release-group': {'id': f'g{i}'}}
                               for i, (t, d) in enumerate(releases)]
        return rec

    def _year(self, recording, group_dates):
        groups = {f'g{i}': d for i, d in enumerate(group_dates)}

        def group(group_id):
            date = groups.get(group_id)
            return {'release-group': {'first-release-date': date} if date else {}}
        with patch('musicbrainzngs.search_recordings',
                   return_value={'recording-count': 1, 'recording-list': [recording]}), \
                patch('musicbrainzngs.get_release_group_by_id', side_effect=group) as mock_group, \
                patch('musicbrainzngs.get_image_front', return_value=None):
            info = search_musicbrainz('Wanted Title', 'Wanted Artist')
        return info.year, [c.args[0] for c in mock_group.call_args_list]

    def test_earliest_release_not_first_listed(self):
        # The first listed release is a later box set (its group starts then, too)
        recording = self._recording(('Box Set', '1994'), ('Album', '1989-03-21'), ('Best Of', '1992'))
        year, groups = self._year(recording, ['1994', '1989-03-21', '1990'])
        self.assertEqual(year, '1989')
        self.assertEqual(groups, ['g1'])  # the group of the earliest release

    def test_release_group_knows_an_earlier_date(self):
        # Only a reissue is listed; its release group starts with the original
        year, _ = self._year(self._recording(('Reissue', '2005-06')), ['1989'])
        self.assertEqual(year, '1989')

    def test_undated_releases_use_the_first_release_group(self):
        year, groups = self._year(self._recording(('A', ''), ('B', None)), ['2001-01-01', '1999'])
        self.assertEqual((year, groups), ('2001', ['g0']))

    def test_no_date_anywhere(self):
        year, _ = self._year(self._recording(('A', '')), [None])
        self.assertIsNone(year)

    def test_official_release_dates_the_song(self):
        recording = self._recording(('Leaked Demo', '1988'), ('Album', '1989-03-21'), ('Box Set', '1994'))
        recording['release-list'][0]['status'] = 'Bootleg'
        recording['release-list'][1]['status'] = 'Official'
        recording['release-list'][2]['status'] = 'Bootleg'
        year, groups = self._year(recording, ['1988', '1989-03-21', '1994'])
        self.assertEqual((year, groups), ('1989', ['g1']))

    def test_undated_official_release_beats_a_dated_bootleg(self):
        recording = self._recording(('Bootleg', '1985'), ('Album', ''))
        recording['release-list'][0]['status'] = 'Bootleg'
        recording['release-list'][1]['status'] = 'Official'
        year, groups = self._year(recording, ['1985', '1989-03-21'])
        self.assertEqual((year, groups), ('1989', ['g1']))  # the official release's group

    def test_year_of(self):
        from src.modules.musicbrainz_client import _year_of
        self.assertEqual([_year_of(d) for d in ('1989', '1989-03', ' 1989-03-21 ', '', None, '19xx')],
                         ['1989', '1989', '1989', None, None, None])


class TestCoverRelease(unittest.TestCase):
    """The cover comes from the earliest official release, not from the first listed one."""

    def test_cover_of_the_earliest_official_release(self):
        rec = _recording('Wanted Title', 'Wanted Artist')
        rec['release-list'] = [
            {'id': 'box', 'status': 'Bootleg', 'date': '1994', 'release-group': {'id': 'g0'}},
            {'id': 'reissue', 'status': 'Official', 'date': '2021-03-05', 'release-group': {'id': 'g1'}},
            {'id': 'single', 'status': 'Official', 'date': '1989-03', 'release-group': {'id': 'g2'}},
        ]
        tried = []

        def front(release_id):
            tried.append(release_id)
            return f'image of {release_id}'.encode()
        with patch('musicbrainzngs.search_recordings',
                   return_value={'recording-count': 1, 'recording-list': [rec]}), \
                patch('musicbrainzngs.get_release_group_by_id', return_value=None), \
                patch('musicbrainzngs.get_image_front', side_effect=front), \
                patch('musicbrainzngs.get_image_list',
                      return_value={'images': [{'front': True, 'image': 'https://example.com/c.jpg'}]}):
            info = search_musicbrainz('Wanted Title', 'Wanted Artist')
        self.assertEqual(info.cover_image_data, b'image of single')
        self.assertEqual(tried, ['single'])

    def test_preferred_order(self):
        from src.modules.musicbrainz_client import _preferred_releases
        releases = [{'id': 'a', 'status': 'Bootleg', 'date': '1980'},
                    {'id': 'b', 'status': 'Official'},
                    {'id': 'c', 'status': 'Official', 'date': '1995'},
                    {'id': 'd', 'status': 'Official', 'date': '1989-03'},
                    {'id': 'e', 'date': '1990'}]
        self.assertEqual([r['id'] for r in _preferred_releases(releases)], ['d', 'c', 'b', 'a', 'e'])

    def test_original_release_before_a_compilation_of_the_same_year(self):
        # A compilation dated only "1994" sorts before "1994-01-10" by date alone
        from src.modules.musicbrainz_client import _preferred_releases
        releases = [{'id': 'hits', 'status': 'Official', 'date': '1994',
                     'release-group': {'type': 'Compilation', 'secondary-type-list': ['Compilation']}},
                    {'id': 'live', 'status': 'Official', 'date': '1993', 'release-group': {'type': 'Live'}},
                    {'id': 'single', 'status': 'Official', 'date': '1994-01-10', 'release-group': {'type': 'Single'}},
                    {'id': 'album', 'status': 'Official', 'date': '1994-01-31', 'release-group': {'type': 'Album'}}]
        self.assertEqual([r['id'] for r in _preferred_releases(releases)], ['single', 'album', 'live', 'hits'])


class TestTitleAdditions(unittest.TestCase):
    """Additions of a video title neither prevent the match nor pick a version of the song."""

    @staticmethod
    def _recording(title, rec_id, *releases):
        rec = _recording(title, 'Wanted Artist')
        rec['id'] = rec_id
        rec['release-list'] = [{'id': f'{rec_id}-{i}', 'status': status, 'date': date,
                                'release-group': {'id': f'{rec_id}-g{i}', 'type': kind}}
                               for i, (status, date, kind) in enumerate(releases)]
        return rec

    def _search(self, title, *recordings):
        with patch('musicbrainzngs.search_recordings',
                   return_value={'recording-count': len(recordings), 'recording-list': list(recordings)}) as search, \
                patch('musicbrainzngs.get_release_group_by_id', return_value=None), \
                patch('musicbrainzngs.get_image_front', return_value=None):
            info = search_musicbrainz(title, 'Wanted Artist')
        return info, search

    def test_additions_do_not_prevent_the_match(self):
        info, search = self._search('Wanted Title - US Version (Official Video)',
                                    self._recording('Wanted Title', 'r1', ('Official', '2006', 'Single')))
        self.assertEqual((info.title, info.year), ('Wanted Title', '2006'))
        self.assertEqual(search.call_args_list[0].kwargs['recording'], 'wanted title')
        # recordings on official releases rank higher among the many of a famous song
        self.assertEqual(search.call_args_list[0].kwargs['status'], 'official')

    def test_title_without_additions_and_year_of_all_recordings(self):
        # The search lists a video recording first; the song came out years before
        info, _ = self._search('Wanted Title (US Version) (Official Music Video)',
                               self._recording('Wanted Title (US)', 'video', ('Official', '2006-02-14', 'Video')),
                               self._recording('Wanted Title', 'live', ('Bootleg', '1994-05', 'Live')),
                               self._recording('Wanted Title', 'studio', ('Official', '1994-01-31', 'Album')))
        self.assertEqual((info.title, info.year), ('Wanted Title', '1994'))

    def test_unknown_bracketed_addition_still_matches(self):
        info, _ = self._search('Wanted Title (Some Channel Words)',
                               self._recording('Wanted Title', 'r1', ('Official', '2006', 'Single')))
        self.assertEqual(info.title, 'Wanted Title')

    def test_matching_bracketed_part_first(self):
        info, _ = self._search('Some Song (Part 2)',
                               self._recording('Some Song', 'one', ('Official', '2001', 'Single')),
                               self._recording('Some Song (Part 2)', 'two', ('Official', '2005', 'Single')))
        self.assertEqual((info.title, info.year), ('Some Song (Part 2)', '2005'))

    def test_other_part_does_not_date_the_song(self):
        # Without a match of the unknown bracketed part, the wanted title is compared without it;
        # the recordings keep their parts, so "(Part 2)" (listed first) is no match
        info, _ = self._search('Some Song (Some Channel Words)',
                               self._recording('Some Song (Part 2)', 'two', ('Official', '1999', 'Single')),
                               self._recording('Some Song', 'one', ('Official', '2001', 'Single')))
        self.assertEqual((info.title, info.year), ('Some Song', '2001'))

    def test_other_part_is_no_match(self):
        for wanted, listed in (('Some Song (Part 1)', 'Some Song (Part 2)'), ('Some Song (Part 2)', 'Some Song')):
            with self.subTest(wanted=wanted, listed=listed):
                info, _ = self._search(wanted, self._recording(listed, 'r1', ('Official', '1999', 'Single')))
                self.assertEqual((info.title, info.year), (wanted, None))

    def test_other_part_is_no_match_with_artist_and_title_swapped(self):
        # File "Some Song (Part 2) - Wanted Artist": the song comes as the artist
        recording = self._recording('Some Song', 'r1', ('Official', '1999', 'Single'))
        with patch('musicbrainzngs.search_recordings',
                   return_value={'recording-count': 1, 'recording-list': [recording]}), \
                patch('musicbrainzngs.get_release_group_by_id', return_value=None), \
                patch('musicbrainzngs.get_image_front', return_value=None):
            info = search_musicbrainz('Wanted Artist', 'Some Song (Part 2)')
        self.assertIsNone(info.year)

    def test_description_after_a_dash(self):
        # "Wanted Title - Singer Name live in the studio": nothing has that title, the song is found by
        # the title before the dash
        recording = self._recording('Wanted Title', 'r1', ('Official', '2019', 'Single'))

        def search(**kwargs):
            found = kwargs['recording'].lower() == 'wanted title'
            return {'recording-count': int(found), 'recording-list': [recording] if found else []}

        out = io.StringIO()
        with patch('musicbrainzngs.search_recordings', side_effect=search) as mock_search, \
                patch('musicbrainzngs.get_release_group_by_id', return_value=None), \
                patch('musicbrainzngs.get_image_front', return_value=None), redirect_stdout(out):
            info = search_musicbrainz('Wanted Title - Singer Name live in the studio', 'Wanted Artist')
        self.assertEqual((info.title, info.artist, info.year), ('Wanted Title', 'Wanted Artist', '2019'))
        self.assertEqual([c.kwargs['recording'].lower() for c in mock_search.call_args_list][::2],
                         ['wanted title - singer name live in the studio', 'wanted title'])
        # reported as written
        self.assertIn('Searching again with the title before the dash', out.getvalue())
        self.assertIn('Wanted Title', out.getvalue())

    def test_description_after_a_dash_quick(self):
        # the GUI's lyrics badge: the same search, nothing reported
        out = io.StringIO()
        with patch('musicbrainzngs.search_recordings',
                   return_value={'recording-count': 0, 'recording-list': []}) as mock_search, redirect_stdout(out):
            info = search_musicbrainz('Wanted Title - Singer Name live in the studio', 'Wanted Artist', quick=True)
        self.assertEqual(info.title, 'Wanted Title - Singer Name live in the studio')
        self.assertEqual(mock_search.call_count, 4)
        self.assertEqual(out.getvalue(), '')

    def test_description_is_not_cut_when_the_title_matches(self):
        recording = self._recording('Wanted Title - Singer Name live in the studio', 'r1',
                                    ('Official', '2019', 'Single'))
        info, search = self._search('Wanted Title - Singer Name live in the studio', recording)
        self.assertEqual(info.title, 'Wanted Title - Singer Name live in the studio')
        self.assertEqual(search.call_count, 2)  # artist + title, and swapped

    def test_part_after_a_dash_is_not_cut(self):
        recording = self._recording('Wanted Title', 'r1', ('Official', '2019', 'Single'))
        info, search = self._search('Wanted Title - Part 2 of the Story', recording)
        self.assertEqual((info.title, info.year), ('Wanted Title - Part 2 of the Story', None))
        self.assertEqual(search.call_count, 2)

    def test_each_release_group_asked_once(self):
        with patch('musicbrainzngs.search_recordings',
                   return_value={'recording-count': 1, 'recording-list': [
                       self._recording('Wanted Title', 'r1', ('Official', '', 'Album'))]}), \
                patch('musicbrainzngs.get_release_group_by_id', return_value=None) as group, \
                patch('musicbrainzngs.get_image_front', return_value=None):
            info = search_musicbrainz('Wanted Title', 'Wanted Artist')
        self.assertIsNone(info.year)
        self.assertEqual(group.call_count, 1)

    def test_year_of_a_compilation_when_the_album_has_no_date(self):
        info, _ = self._search('Wanted Title',
                               self._recording('Wanted Title', 'album', ('Official', '', 'Album')),
                               self._recording('Wanted Title', 'hits', ('Official', '1990', 'Compilation')))
        self.assertEqual(info.year, '1990')

    def test_only_recordings_of_the_same_title_date_the_song(self):
        # A similar title of the same artist is another song: it must not lend its year
        info, _ = self._search('Wanted Title',
                               self._recording('Wanted Title', 'song', ('Official', '2010', 'Album')),
                               self._recording('Wanted Titles', 'other', ('Official', '1995', 'Single')))
        self.assertEqual((info.title, info.year), ('Wanted Title', '2010'))

    def test_live_and_compilation_releases_do_not_date_the_song(self):
        info, _ = self._search('Wanted Title',
                               self._recording('Wanted Title', 'studio', ('Official', '1994-01-31', 'Album')),
                               self._recording('Wanted Title (live)', 'live', ('Official', '1993', 'Live')),
                               self._recording('Wanted Title', 'hits', ('Official', '1992', 'Compilation')))
        self.assertEqual(info.year, '1994')

    def test_later_kinds(self):
        from src.modules.musicbrainz_client import _is_later_kind
        for group, later in (({'type': 'Album', 'secondary-type-list': ['Compilation']}, True),
                             ({'type': 'Compilation'}, True), ({'type': 'Live'}, True),
                             ({'type': 'Album'}, False), ({'type': 'Single'}, False), (None, False), ({}, False)):
            with self.subTest(group=group):
                self.assertEqual(_is_later_kind({'release-group': group}), later)

    def test_at_most_some_releases_are_asked_for_a_cover(self):
        from src.modules.musicbrainz_client import MAX_COVER_RELEASES
        recording = self._recording('Wanted Title', 'r1', *[('Official', f'20{i:02d}', 'Single') for i in range(20)])
        with patch('musicbrainzngs.search_recordings',
                   return_value={'recording-count': 1, 'recording-list': [recording]}), \
                patch('musicbrainzngs.get_release_group_by_id', return_value=None), \
                patch('musicbrainzngs.get_image_front', return_value=None) as front:
            search_musicbrainz('Wanted Title', 'Wanted Artist')
        self.assertEqual(front.call_count, MAX_COVER_RELEASES)

    def test_releases_of_all_recordings_once(self):
        from src.modules.musicbrainz_client import _releases_of
        shared = {'id': 'album'}
        recordings = [{'release-list': [shared, {'id': 'single'}]}, {'release-list': [dict(shared)]}, {}]
        self.assertEqual([r['id'] for r in _releases_of(recordings)], ['album', 'single'])


class TestTitleBeforeAnyDash(unittest.TestCase):
    """Last, with the length of the audio known, the title before a dash is searched whatever follows it
    ("Wanted Title - Chapter 2"), for a recording of that length: "Wanted Title - Part 2" can be another
    song than "Wanted Title"."""

    @staticmethod
    def _recording(title, rec_id, length, year='2019', artist='Wanted Artist'):
        """A recording of a single from ``year``; its ``length`` in milliseconds, as the song database
        gives it (None: unknown)."""
        rec = TestTitleAdditions._recording(title, rec_id, ('Official', year, 'Single'))
        rec['artist-credit-phrase'] = artist
        if length is not None:
            rec['length'] = length
        return rec

    def _search(self, title, recordings, duration=None, quick=False):
        """The song found for ``title``, the titles searched and the console. Like a search engine, the
        song database lists the ``recordings`` for any search that names "Wanted Title"; the covers
        asked for go to ``self.covers``."""
        def search(**kwargs):
            found = recordings if 'wanted title' in kwargs['recording'].lower() else []
            return {'recording-count': len(found), 'recording-list': found}

        out = io.StringIO()
        with patch('musicbrainzngs.search_recordings', side_effect=search) as mock_search, \
                patch('musicbrainzngs.get_release_group_by_id', return_value=None), \
                patch('musicbrainzngs.get_image_front', return_value=None) as front, redirect_stdout(out):
            info = search_musicbrainz(title, 'Wanted Artist', quick=quick, duration=duration)
        self.covers = [c.args[0] for c in front.call_args_list]
        # Each title is searched as the title and, swapped, as the artist
        return info, [c.kwargs['recording'].lower() for c in mock_search.call_args_list][::2], out.getvalue()

    def test_a_recording_of_the_length_of_the_audio(self):
        info, searched, out = self._search('Wanted Title [OFFICIAL VIDEO] - Chapter 2',
                                           [self._recording('Wanted Title', 'r1', '239000')], duration=238.0)
        self.assertEqual((info.title, info.artist, info.year), ('Wanted Title', 'Wanted Artist', '2019'))
        self.assertEqual(searched, ['wanted title - chapter 2', 'wanted title'])
        self.assertIn('Searching again with the title before the dash, for a recording of the length of the audio',
                      out)

    def test_up_to_5_s_off_the_length_of_the_audio(self):
        for length, taken in (('243000', True), ('233000', True), ('243010', False), ('', False), (None, False)):
            with self.subTest(length=length):
                info, _, _ = self._search('Wanted Title - Part 2', [self._recording('Wanted Title', 'r1', length)],
                                          duration=238.0)
                self.assertEqual(info.title, 'Wanted Title' if taken else 'Wanted Title - Part 2')

    def test_only_with_the_length_of_the_audio(self):
        info, searched, _ = self._search('Wanted Title - Chapter 2', [self._recording('Wanted Title', 'r1', '238000')])
        self.assertEqual(info.title, 'Wanted Title - Chapter 2')
        self.assertEqual(searched, ['wanted title - chapter 2'])

    def test_the_searches_before_take_any_length(self):
        """The title, and the title before a description, still find a recording of any length."""
        for title, searched in (('Wanted Title', ['wanted title']),
                                ('Wanted Title - Singer Name live in the studio',
                                 ['wanted title - singer name live in the studio', 'wanted title'])):
            for length in ('300000', None):
                with self.subTest(title=title, length=length):
                    info, asked, _ = self._search(title, [self._recording('Wanted Title', 'r1', length)],
                                                  duration=238.0)
                    self.assertEqual(info.title, 'Wanted Title')
                    self.assertEqual(asked, searched)

    def test_a_title_is_searched_once(self):
        # Before a description, the title before the dash was searched already
        _, searched, _ = self._search('Wanted Title - Singer Name live in the studio', [], duration=238.0)
        self.assertEqual(searched, ['wanted title - singer name live in the studio', 'wanted title'])

    def test_only_recordings_of_the_length_of_the_audio_name_and_date_the_song(self):
        # Listed first, a recording of another length is another song: not taken, and without its
        # year and cover
        recordings = [self._recording('Wanted Title', 'other', '180000', year='2005'),
                      self._recording('Wanted Title', 'song', '239000', artist='Wanted Artist feat. Guest')]
        info, _, _ = self._search('Wanted Title - Chapter 2', recordings, duration=238.0)
        self.assertEqual((info.artist, info.title, info.year), ('Wanted Artist feat. Guest', 'Wanted Title', '2019'))
        self.assertEqual(self.covers, ['song-0'])

    def test_a_bracketed_part_of_another_length(self):
        # The bracketed part has to match first, but a recording of another length does not count
        info, _, _ = self._search('Wanted Title (Some Words) - Chapter 2',
                                  [self._recording('Wanted Title (Some Words)', 'other', '180000'),
                                   self._recording('Wanted Title', 'song', '239000')], duration=238.0)
        self.assertEqual(info.title, 'Wanted Title')

    def test_quick(self):
        """The GUI's lyrics badge names the song so too, reporting nothing."""
        recordings = [self._recording('Wanted Title', 'r1', '239000')]
        full, full_searched, _ = self._search('Wanted Title [OFFICIAL VIDEO] - Chapter 2', recordings, duration=238.0)
        quick, quick_searched, out = self._search('Wanted Title [OFFICIAL VIDEO] - Chapter 2', recordings,
                                                  duration=238.0, quick=True)
        self.assertEqual((quick.artist, quick.title), (full.artist, full.title))
        self.assertEqual((quick.artist, quick.title, quick.year), ('Wanted Artist', 'Wanted Title', None))
        self.assertEqual(quick_searched, full_searched)
        self.assertEqual((self.covers, out), ([], ''))


class TestSimilarity(unittest.TestCase):
    def test_values(self):
        from src.modules.name_matching import similarity as _similarity
        self.assertEqual(_similarity('Some Title', 'some title!'), 1.0)
        self.assertGreaterEqual(_similarity('Title', 'Title (Live) Full HD'), 0.8)
        self.assertLess(_similarity('Long Artist Name', 'Short'), 0.8)
        self.assertEqual(_similarity('', 'x'), 0.0)


class TestDistinctNamesSharingAWord(unittest.TestCase):
    """A shared word does not make two different artists or titles the same."""

    @patch('musicbrainzngs.search_recordings')
    def test_artist_with_extra_name_word_rejected(self, mock_search):
        mock_search.return_value = {'recording-count': 1,
                                    'recording-list': [_recording('Wanted Title', 'Nova Lights')]}
        info = search_musicbrainz('Wanted Title', 'Nova')
        self.assertEqual((info.artist, info.title), ('Nova', 'Wanted Title'))

    def test_similarity_rules(self):
        from src.modules.name_matching import similarity as _similarity
        self.assertLess(_similarity('Nova Lights', 'Nova', artist=True), 0.8)
        self.assertLess(_similarity('Love', 'Love Me Tender'), 0.8)
        self.assertGreaterEqual(_similarity('Title', 'Title Full HD 2023'), 0.8)        # qualifiers only
        self.assertEqual(_similarity('Title', 'Title (Some Other Words)'), 1.0)          # brackets ignored
        self.assertEqual(_similarity('Nova feat. Guest', 'Nova', artist=True), 1.0)      # featured guest ignored
        self.assertLess(_similarity('Nova & Friends', 'Nova', artist=True), 0.8)         # a band name is not a feature


class TestNumberedTitles(unittest.TestCase):
    def test_different_numbers_are_different_songs(self):
        from src.modules.name_matching import similarity as _similarity
        self.assertEqual(_similarity('Song 2', 'Song 3'), 0.0)
        self.assertEqual(_similarity('Part 1', 'Part 10'), 0.0)
        self.assertGreaterEqual(_similarity('Title', 'Title 3'), 0.8)   # one-sided number still a match
        self.assertEqual(_similarity('Song 2', 'Song 2'), 1.0)

    @patch('musicbrainzngs.search_recordings')
    def test_wrong_numbered_title_rejected(self, mock_search):
        mock_search.return_value = {'recording-count': 1,
                                    'recording-list': [_recording('Wanted Part 3', 'Wanted Artist')]}
        info = search_musicbrainz('Wanted Part 2', 'Wanted Artist')
        self.assertEqual(info.title, 'Wanted Part 2')


def _response_error(status):
    """What musicbrainzngs raises for HTTP 400, 404 and 411."""
    return musicbrainzngs.ResponseError(cause=HTTPError("u", status, "msg", None, None))


class TestNotFoundAnswers(unittest.TestCase):
    """A request that MusicBrainz answers with "not found" or "bad request" has no result: no crash."""

    def test_merged_release_group(self):
        # The search still lists a release group that was merged into another one
        recording = TestYear._recording(('Album', '1989-03-21'))
        with patch('musicbrainzngs.search_recordings',
                   return_value={'recording-count': 1, 'recording-list': [recording]}), \
                patch('musicbrainzngs.get_release_group_by_id', side_effect=_response_error(404)), \
                patch('musicbrainzngs.get_image_front', return_value=None):
            info = search_musicbrainz('Wanted Title', 'Wanted Artist')
        self.assertEqual((info.title, info.year), ('Wanted Title', '1989'))

    def test_bad_request(self):
        with patch('musicbrainzngs.search_recordings', side_effect=_response_error(400)):
            info = search_musicbrainz('Wanted Title', 'Wanted Artist')
        self.assertEqual((info.artist, info.title), ('Wanted Artist', 'Wanted Title'))


class _Clock:
    """Stand-in for the time module of musicbrainz_client: no waiting, and a clock that only the test moves."""

    def __init__(self):
        self.now = 0.0

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class TestRequestLimits(unittest.TestCase):
    """Requests time out, and a lookup stops asking once its failed requests took too long."""

    def setUp(self):
        self.clock = _Clock()
        patcher = patch.object(musicbrainz_client, "time", self.clock)
        patcher.start()
        self.addCleanup(patcher.stop)
        musicbrainz_client._lookup.failed_seconds = 0.0
        self.addCleanup(setattr, musicbrainz_client._lookup, "failed_seconds", 0.0)

    def _overloaded(self, *args, **kwargs):
        # musicbrainzngs gives up on a request to an overloaded server after about a minute
        self.clock.now += 56
        raise musicbrainzngs.NetworkError("retried 8 times")

    def test_requests_have_a_timeout(self):
        timeouts = []

        def search(*args, **kwargs):
            timeouts.append(socket.getdefaulttimeout())
            return {'recording-count': 0, 'recording-list': []}
        before = socket.getdefaulttimeout()
        with patch('musicbrainzngs.search_recordings', side_effect=search):
            search_musicbrainz('Wanted Title', 'Wanted Artist')
        self.assertEqual(timeouts, [musicbrainz_client.REQUEST_TIMEOUT_SECONDS] * 2)
        self.assertEqual(socket.getdefaulttimeout(), before)

    def test_cover_search_stops_when_the_server_does_not_answer(self):
        get_image = musicbrainz_client.__dict__["__get_image"]
        releases = [{'id': f'r{i}', 'status': 'Official'} for i in range(15)]
        out = io.StringIO()
        with patch('musicbrainzngs.get_image_front', side_effect=self._overloaded) as front, redirect_stdout(out):
            self.assertEqual(get_image(releases), (None, None))
        # Not 3 tries for each of the 15 releases, 43 minutes
        self.assertEqual(front.call_count, 1)
        self.assertIn("MusicBrainz does not answer", out.getvalue())

    def test_quick_failure_is_tried_again(self):
        request = musicbrainz_client.__dict__["__musicbrainz_request"]
        answers = [musicbrainzngs.NetworkError("connection reset"), {'release-group': {}}]
        with patch('musicbrainzngs.get_release_group_by_id', side_effect=answers) as group:
            self.assertEqual(request(lambda: musicbrainzngs.get_release_group_by_id('g')), {'release-group': {}})
        self.assertEqual(group.call_count, 2)

    def test_each_lookup_starts_afresh(self):
        with patch('musicbrainzngs.search_recordings', side_effect=self._overloaded) as search:
            search_musicbrainz('Wanted Title', 'Wanted Artist')
            self.assertEqual(search.call_count, 1)  # the second search of the lookup is not made
            search_musicbrainz('Wanted Title', 'Wanted Artist')
        self.assertEqual(search.call_count, 2)

    def test_overlapping_requests_keep_the_timeout(self):
        """Requests of lookups in several threads (the GUI's lyrics badge) overlap: the timeout
        stays set until the last one has ended, then the default timeout before comes back."""
        before = socket.getdefaulttimeout()
        second = musicbrainz_client._socket_timeout(musicbrainz_client.REQUEST_TIMEOUT_SECONDS)
        with musicbrainz_client._socket_timeout(musicbrainz_client.REQUEST_TIMEOUT_SECONDS):
            second.__enter__()
        try:
            # The first request has ended, the second one still runs
            self.assertEqual(socket.getdefaulttimeout(), musicbrainz_client.REQUEST_TIMEOUT_SECONDS)
        finally:
            second.__exit__(None, None, None)
        self.assertEqual(socket.getdefaulttimeout(), before)

    def test_a_lookup_in_another_thread_does_not_use_up_the_time_of_this_one(self):
        """The GUI's lyrics badge looks songs up in several threads at a time."""
        this_thread = threading.current_thread()
        searches = []

        def search(*args, **kwargs):
            searches.append(threading.current_thread())
            if threading.current_thread() is not this_thread:
                self._overloaded()
            if len(searches) == 1:
                # Meanwhile a lookup in another thread finds the song database overloaded
                other = threading.Thread(target=search_musicbrainz, args=('Other Title', 'Other Artist'))
                other.start()
                other.join()
            return {'recording-count': 0, 'recording-list': []}
        with patch('musicbrainzngs.search_recordings', side_effect=search), redirect_stdout(io.StringIO()):
            search_musicbrainz('Wanted Title', 'Wanted Artist')
        # Both searches of this lookup are made; the other lookup gave up after its first
        self.assertEqual(searches.count(this_thread), 2)
        self.assertEqual(len(searches), 3)


class TestQuickLookup(unittest.TestCase):
    """The GUI's lyrics badge names the song as a conversion does, without the year, genres
    and cover, and without reporting anything."""

    def setUp(self):
        self.clock = _Clock()
        patcher = patch.object(musicbrainz_client, "time", self.clock)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _overloaded(self, *args, **kwargs):
        self.clock.now += 56
        raise musicbrainzngs.NetworkError("retried 8 times")

    def _search(self, quick, failure=None):
        recording = TestYear._recording(('Album', '1989-03-21'))
        recording['tag-list'] = [{'name': 'Genre'}]
        out = io.StringIO()
        with patch('musicbrainzngs.search_recordings',
                   return_value={'recording-count': 1, 'recording-list': [recording]},
                   side_effect=failure) as search, \
                patch('musicbrainzngs.get_release_group_by_id', return_value=None) as group, \
                patch('musicbrainzngs.get_image_front', return_value=None) as front, redirect_stdout(out):
            info = search_musicbrainz('Wanted Title (Official Video)', 'WANTED ARTIST', quick=quick)
        return info, search, group.call_count + front.call_count, out.getvalue()

    def test_same_song_name_as_the_full_lookup(self):
        full, full_search, _, _ = self._search(quick=False)
        quick, quick_search, details, out = self._search(quick=True)
        self.assertEqual((quick.artist, quick.title), (full.artist, full.title))
        self.assertEqual((quick.artist, quick.title), ('Wanted Artist', 'Wanted Title'))
        self.assertEqual(quick_search.call_args_list, full_search.call_args_list)
        self.assertEqual((quick.year, quick.genres, quick.cover_image_data), (None, None, None))
        self.assertEqual(details, 0)
        self.assertEqual(out, "")

    def test_failures_are_not_reported(self):
        for failure in (_response_error(400), self._overloaded):
            with self.subTest(failure=failure):
                _, _, _, full_out = self._search(quick=False, failure=failure)
                self.assertNotEqual(full_out, "")  # what a conversion reports
                info, _, _, out = self._search(quick=True, failure=failure)
                self.assertEqual((info.artist, info.title), ('WANTED ARTIST', 'Wanted Title (Official Video)'))
                self.assertEqual(out, "")
