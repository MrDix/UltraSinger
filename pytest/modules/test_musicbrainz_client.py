"""Test the musicbrainz_client module."""

import unittest
from unittest.mock import patch
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
        recording = {"release-list": [{"id": "r1"}]}
        buf = _io.StringIO()
        with patch.object(mb.musicbrainzngs, "get_image_front",
                          side_effect=exc), redirect_stdout(buf):
            result = get_image(recording)
        return result, buf.getvalue()

    def test_404_cause_is_silent(self):
        result, out = self._run(404)
        self.assertEqual(result, (None, None))
        self.assertNotIn("Cover art download failed", out)
        self.assertIn("No cover art on MusicBrainz", out)

    def test_other_status_is_loud(self):
        _, out = self._run(503)
        self.assertIn("Cover art download failed", out)


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
        def search(recording=None, limit=None, artist=None, artistname=None):
            if artist == 'wanted artist':
                return {'recording-count': 1, 'recording-list': [_recording('Wanted Title', 'Wanted Artist')]}
            return {'recording-count': 0, 'recording-list': []}
        mock_search.side_effect = search
        info = search_musicbrainz('Wanted Artist', 'Wanted Title')
        self.assertEqual((info.artist, info.title), ('Wanted Artist', 'Wanted Title'))


class TestSimilarity(unittest.TestCase):
    def test_values(self):
        from src.modules.musicbrainz_client import _similarity
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
        from src.modules.musicbrainz_client import _similarity
        self.assertLess(_similarity('Nova Lights', 'Nova', artist=True), 0.8)
        self.assertLess(_similarity('Love', 'Love Me Tender'), 0.8)
        self.assertGreaterEqual(_similarity('Title', 'Title Full HD 2023'), 0.8)        # qualifiers only
        self.assertEqual(_similarity('Title', 'Title (Some Other Words)'), 1.0)          # brackets ignored
        self.assertEqual(_similarity('Nova feat. Guest', 'Nova', artist=True), 1.0)      # featured guest ignored
        self.assertLess(_similarity('Nova & Friends', 'Nova', artist=True), 0.8)         # a band name is not a feature


class TestNumberedTitles(unittest.TestCase):
    def test_different_numbers_are_different_songs(self):
        from src.modules.musicbrainz_client import _similarity
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
