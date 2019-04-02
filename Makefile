
TARGETS = $(wildcard cmd/*)

# ------------------------------------------------------------------------------

all:
	@for i in $(TARGETS); do (cd $$i && go install); done

install: all
