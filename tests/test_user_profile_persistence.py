import os
import tempfile
import unittest
from unittest.mock import patch

import backend.app as app


class UserProfilePersistenceTests(unittest.TestCase):
    def test_public_user_includes_normalized_default_profile(self):
        public = app._to_public_user({"user_id": "u1", "email": "a@example.com"})

        self.assertEqual(public["user_profile"]["risk_profile"], "balanced")
        self.assertEqual(public["user_profile"]["holding_period"], "swing")
        self.assertEqual(public["user_profile"]["max_position_per_stock"], 20.0)
        self.assertEqual(public["avatar_url"], "")

    def test_update_user_profile_persists_to_users_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            users_path = os.path.join(tmpdir, "users.json")
            app.write_json(
                users_path,
                [
                    {
                        "user_id": "u1",
                        "email": "a@example.com",
                        "phone": "13800138000",
                        "nickname": "tester",
                    }
                ],
            )

            with patch.object(app, "USERS_PATH", users_path):
                profile = app._update_user_profile(
                    "u1",
                    {
                        "risk_profile": "aggressive",
                        "holding_period": "mid_term",
                        "max_position_per_stock": 35,
                        "already_holding": True,
                        "current_position_percent": 22,
                        "cost_price": 18.5,
                        "prefer_stop_loss": False,
                    },
                )
                users = app._load_users()

        self.assertEqual(profile["risk_profile"], "aggressive")
        self.assertEqual(profile["holding_period"], "mid_term")
        self.assertEqual(users[0]["user_profile"], profile)
        self.assertTrue(users[0]["profile_updated_at"])

    def test_update_user_account_persists_public_fields(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            users_path = os.path.join(tmpdir, "users.json")
            app.write_json(
                users_path,
                [
                    {
                        "user_id": "u1",
                        "email": "old@example.com",
                        "phone": "13800138000",
                        "nickname": "old",
                    }
                ],
            )

            with patch.object(app, "USERS_PATH", users_path):
                public = app._update_user_account(
                    "u1",
                    email="new@example.com",
                    phone="13900139000",
                    nickname="newname",
                )
                users = app._load_users()

        self.assertEqual(public["email"], "new@example.com")
        self.assertEqual(public["phone"], "13900139000")
        self.assertEqual(public["nickname"], "newname")
        self.assertEqual(users[0]["email"], "new@example.com")
        self.assertTrue(users[0]["account_updated_at"])

    def test_update_user_avatar_persists_public_url(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            users_path = os.path.join(tmpdir, "users.json")
            app.write_json(
                users_path,
                [
                    {
                        "user_id": "u1",
                        "email": "a@example.com",
                        "phone": "13800138000",
                        "nickname": "tester",
                    }
                ],
            )

            with patch.object(app, "USERS_PATH", users_path):
                public = app._update_user_avatar("u1", "/avatars/u1.png")
                users = app._load_users()

        self.assertEqual(public["avatar_url"], "/avatars/u1.png")
        self.assertEqual(users[0]["avatar_url"], "/avatars/u1.png")
        self.assertTrue(users[0]["avatar_updated_at"])

    def test_update_stock_personalization_persists_by_user_and_ticker(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            users_path = os.path.join(tmpdir, "users.json")
            personalization_path = os.path.join(tmpdir, "user_stock_personalization.json")
            app.write_json(
                users_path,
                [
                    {
                        "user_id": "u1",
                        "email": "a@example.com",
                        "phone": "13800138000",
                        "nickname": "tester",
                        "user_profile": {"risk_profile": "balanced", "max_position_per_stock": 20},
                    }
                ],
            )

            with (
                patch.object(app, "USERS_PATH", users_path),
                patch.object(app, "USER_STOCK_PERSONALIZATION_PATH", personalization_path),
            ):
                saved = app._update_user_stock_personalization(
                    "u1",
                    "000001",
                    {"risk_profile": "conservative", "max_position_per_stock": 12},
                    {
                        "use_system_rules": False,
                        "custom_rules": [{"policy": "block_buy", "text": "只在放量回踩后买入"}],
                        "custom_constraints": [{"type": "block_buy", "text": "财报前不加仓"}],
                    },
                )
                loaded = app._get_user_stock_personalization("u1", "000001")

        self.assertEqual(saved["ticker"], "000001")
        self.assertEqual(loaded["profile"]["risk_profile"], "conservative")
        self.assertEqual(loaded["profile"]["max_position_per_stock"], 12.0)
        self.assertFalse(loaded["preferences"]["use_system_rules"])
        self.assertEqual(loaded["preferences"]["custom_rules"][0]["policy"], "block_buy")
        self.assertEqual(loaded["preferences"]["custom_constraints"][0]["type"], "block_buy")

    def test_change_user_password_rehashes_password(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            users_path = os.path.join(tmpdir, "users.json")
            salt = "salt1"
            old_hash = app._hash_password("oldpass", salt)
            app.write_json(
                users_path,
                [
                    {
                        "user_id": "u1",
                        "email": "a@example.com",
                        "phone": "13800138000",
                        "password_salt": salt,
                        "password_hash": old_hash,
                    }
                ],
            )

            with patch.object(app, "USERS_PATH", users_path):
                app._change_user_password("u1", "oldpass", "newpass1")
                users = app._load_users()

        self.assertNotEqual(users[0]["password_hash"], old_hash)
        self.assertTrue(app._verify_password("newpass1", users[0]["password_salt"], users[0]["password_hash"]))

    def test_startup_clears_sessions_by_default(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sessions_path = os.path.join(tmpdir, "sessions.json")
            app.write_json(sessions_path, [{"token": "t1", "user_id": "u1"}])

            with patch.object(app, "SESSIONS_PATH", sessions_path), patch.dict(os.environ, {}, clear=True):
                cleared = app._clear_startup_sessions_if_needed()
                sessions = app._load_sessions()

        self.assertTrue(cleared)
        self.assertEqual(sessions, [])


if __name__ == "__main__":
    unittest.main()
