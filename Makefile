run:
	@python3 src/ekf/ekf-localization.py docs/MRCLAM_Dataset1 "Robot1"

install-dependencies:
	pip install -r requirements.txt
