from recommendation_service.infrastructure import database as db
from recommendation_service.repositories.reactions import ReactionRepository
from recommendation_service.repositories.users import UserRepository


def test_reaction_upsert(tmp_path):
    db.init_db(f"sqlite:///{tmp_path / 'test.db'}")
    session = next(db.get_db())
    try:
        user = UserRepository(session).upsert_from_telegram({"id": 100, "first_name": "A"})
        repo = ReactionRepository(session)
        first = repo.upsert(user.id, 10, "like")
        second = repo.upsert(user.id, 10, "dislike")
        assert first.id == second.id
        assert repo.split_profile(user.id) == ([], [10])
        assert repo.delete_for_user(user.id, 10) is True
        assert repo.split_profile(user.id) == ([], [])
        assert repo.delete_for_user(user.id, 10) is False
    finally:
        session.close()
