import pytest

from webapp.users_db import UsersStore, UsernameTaken, EmailTaken, sync_legacy_user, slugify_username


@pytest.fixture()
def store(tmp_path):
    return UsersStore(tmp_path / "users.db")


def test_create_and_lookup(store):
    user = store.create("jane_doe", "Jane Doe", "jane@example.com", sample_count=20)
    assert user.label == 1
    assert store.get_by_username("jane_doe").label == user.label
    assert store.get_by_username("JANE_DOE").label == user.label   # case-insensitive
    assert store.get_by_label(user.label).display_name == "Jane Doe"


def test_labels_increment(store):
    a = store.create("alice", "Alice", None, sample_count=10)
    b = store.create("bob", "Bob", None, sample_count=10)
    assert b.label == a.label + 1


def test_duplicate_username_rejected(store):
    store.create("alice", "Alice", None, sample_count=10)
    with pytest.raises(UsernameTaken):
        store.create("alice", "Someone Else", None, sample_count=10)


def test_duplicate_email_rejected(store):
    store.create("alice", "Alice", "a@example.com", sample_count=10)
    with pytest.raises(EmailTaken):
        store.create("bob", "Bob", "a@example.com", sample_count=10)


def test_username_taken_check(store):
    assert store.username_taken("alice") is False
    store.create("alice", "Alice", None, sample_count=10)
    assert store.username_taken("alice") is True
    assert store.username_taken("ALICE") is True


def test_dataset_dirs_reflects_all_users(store):
    store.create("alice", "Alice", None, sample_count=10)
    store.create("bob", "Bob", None, sample_count=10)
    pairs = dict(store.dataset_dirs())
    assert pairs == {1: "alice", 2: "bob"}


def test_update_profile_changes_name_and_email(store):
    user = store.create("alice", "Alice", "a@example.com", sample_count=10)
    updated = store.update_profile(user.label, "Alice Smith", "alice.smith@example.com")
    assert updated.display_name == "Alice Smith"
    assert updated.email == "alice.smith@example.com"
    assert store.get_by_label(user.label).display_name == "Alice Smith"


def test_update_profile_rejects_email_taken_by_someone_else(store):
    store.create("alice", "Alice", "a@example.com", sample_count=10)
    bob = store.create("bob", "Bob", "b@example.com", sample_count=10)
    with pytest.raises(EmailTaken):
        store.update_profile(bob.label, "Bob", "a@example.com")


def test_update_profile_allows_keeping_own_email(store):
    user = store.create("alice", "Alice", "a@example.com", sample_count=10)
    updated = store.update_profile(user.label, "Alice", "a@example.com")
    assert updated.email == "a@example.com"


def test_delete_removes_user(store):
    user = store.create("alice", "Alice", None, sample_count=10)
    store.delete(user.label)
    assert store.get_by_label(user.label) is None
    assert store.count() == 0


def test_delete_frees_username_for_reuse(store):
    user = store.create("alice", "Alice", None, sample_count=10)
    store.delete(user.label)
    assert store.username_taken("alice") is False
    again = store.create("alice", "Someone New", None, sample_count=5)
    assert again.display_name == "Someone New"   # label may be reallocated (max+1 over an now-empty table); that's fine


def test_slugify():
    assert slugify_username("  Jane_Doe ") == "jane_doe"


def test_sync_legacy_user_imports_once(settings, tmp_path):
    users = UsersStore(tmp_path / "legacy.db")
    sync_legacy_user(users, settings)
    assert users.count() == 1
    first = users.get_by_label(settings.LEGACY_LABEL)
    assert first.username == settings.LEGACY_USERNAME
    assert first.source == "legacy"

    # a second call (e.g. app restart) must not duplicate or error
    sync_legacy_user(users, settings)
    assert users.count() == 1


def test_sync_legacy_user_noop_when_users_exist(settings, tmp_path):
    users = UsersStore(tmp_path / "legacy2.db")
    users.create("someone", "Someone", None, sample_count=5)
    sync_legacy_user(users, settings)
    assert users.count() == 1   # legacy import skipped, only the manual user present


def test_sync_legacy_user_noop_without_dataset(settings_no_legacy, tmp_path):
    users = UsersStore(tmp_path / "legacy3.db")
    sync_legacy_user(users, settings_no_legacy)
    assert users.count() == 0
