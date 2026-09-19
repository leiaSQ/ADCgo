package main

import (
	"bytes"
	"flag"
	"os"
	"strings"
	"sync"
	"testing"
)

// defineFlagsOnce runs main's own flag-definition path exactly once for the whole test
// binary, by invoking main with `-h`: the help intercept sits immediately after the
// flag definitions and returns without solving anything or touching the filesystem.
// Every test below then reads the real flag.CommandLine, so the help pages are checked
// against the flags the binary actually accepts rather than a copy of them. It is a
// sync.Once because a second call would redefine the flags and panic.
var defineFlagsOnce sync.Once

func realFlags(t *testing.T) map[string]*flag.Flag {
	t.Helper()
	defineFlagsOnce.Do(func() {
		devnull, err := os.OpenFile(os.DevNull, os.O_WRONLY, 0)
		if err != nil {
			t.Fatalf("open %s: %v", os.DevNull, err)
		}
		defer devnull.Close()
		stdout, args := os.Stdout, os.Args
		os.Stdout, os.Args = devnull, []string{"adcgo", "-h"}
		main()
		os.Stdout, os.Args = stdout, args
	})

	flags := map[string]*flag.Flag{}
	flag.CommandLine.VisitAll(func(f *flag.Flag) {
		if strings.HasPrefix(f.Name, "test.") { // the testing package's own flags
			return
		}
		flags[f.Name] = f
	})
	if len(flags) == 0 {
		t.Fatal("flag.CommandLine holds no adcgo flags; main's definitions did not run")
	}
	return flags
}

// TestHelpTopicsCoverEveryFlag is the guard that keeps the tiered help honest in both
// directions: every flag the binary defines is reachable from some topic page, and no
// topic names a flag that no longer exists. Without it a new flag would be documented
// nowhere but `adcgo -h all`.
func TestHelpTopicsCoverEveryFlag(t *testing.T) {
	defined := realFlags(t)

	covered := map[string]bool{}
	for _, topic := range helpTopics {
		for _, name := range topic.flags {
			if defined[name] == nil {
				t.Errorf("help topic %q lists -%s, which the binary does not define", topic.key, name)
			}
			covered[name] = true
		}
	}
	for name := range defined {
		if !covered[name] {
			t.Errorf("-%s appears in no help topic; add it to a topic's flags in help.go", name)
		}
	}
}

// TestHelpTopicKeysAreUnique catches a key or alias registered twice, which would make
// the second page unreachable.
func TestHelpTopicKeysAreUnique(t *testing.T) {
	seen := map[string]string{}
	for _, topic := range helpTopics {
		for _, name := range append([]string{topic.key}, topic.aliases...) {
			key := normalizeTopic(name)
			if prev, dup := seen[key]; dup {
				t.Errorf("topic key %q (%s) collides with topic %q", key, name, prev)
			}
			seen[key] = topic.key
		}
	}
}

// TestHelpSeeAlsoResolves keeps the cross-references between pages live.
func TestHelpSeeAlsoResolves(t *testing.T) {
	for _, topic := range helpTopics {
		for _, ref := range topic.see {
			if lookupTopic(normalizeTopic(ref)) == nil {
				t.Errorf("topic %q points at unknown topic %q", topic.key, ref)
			}
		}
	}
}

func TestHelpRequested(t *testing.T) {
	cases := []struct {
		args  []string
		topic string
		want  bool
	}{
		{[]string{"-h"}, "", true},
		{[]string{"--help"}, "", true},
		{[]string{"-help", "sip"}, "sip", true},
		{[]string{"-h", "order", "4"}, "order4", true},
		{[]string{"-h", "lanczos-lowmem"}, "lanczoslowmem", true},
		{[]string{"-h", "solver", "lanczos"}, "lanczos", true},
		{[]string{"-h", "-solver"}, "solver", true}, // a flag name resolves to its page
		{[]string{"-fcidump", "x", "-dip"}, "", false},
		{[]string{"-fcidump", "x", "-h"}, "", true},
	}
	for _, c := range cases {
		topic, ok := helpRequested(c.args)
		if ok != c.want || topic != c.topic {
			t.Errorf("helpRequested(%v) = (%q, %v), want (%q, %v)", c.args, topic, ok, c.topic, c.want)
		}
	}
}

// TestHelpTopicLookup checks the aliases a user is most likely to type.
func TestHelpTopicLookup(t *testing.T) {
	cases := map[string]string{
		"dip": "dip", "sip": "sip", "order": "order", "order22": "order22",
		"adc22": "order22", "cvs": "order4", "adc4": "order4", "adc2x": "order2",
		"lowmem": "lowmem", "lanczoslowmem": "lowmem", "davidson": "davidson",
		"gpu": "backend", "cuda": "backend", "nvlink": "mgpu", "cache": "checkpoint",
		"bare": "spectrum", "rassi": "tdm", "widths": "fano",
	}
	for in, want := range cases {
		got := lookupTopic(normalizeTopic(in))
		if got == nil {
			t.Errorf("lookupTopic(%q) = nil, want %q", in, want)
			continue
		}
		if got.key != want {
			t.Errorf("lookupTopic(%q) = %q, want %q", in, got.key, want)
		}
	}
	if lookupTopic("nosuchtopic") != nil {
		t.Error("lookupTopic resolved a topic that does not exist")
	}
}

// TestPrintHelpRenders exercises the three tiers end to end against the real flag set.
func TestPrintHelpRenders(t *testing.T) {
	realFlags(t)

	var overview bytes.Buffer
	if !printHelp(&overview, "") {
		t.Fatal("printHelp(overview) reported an unknown topic")
	}
	for _, want := range []string{"METHODS", "SOLVERS", "-dip", "-order", "lanczos", "adcgo -h all"} {
		if !strings.Contains(overview.String(), want) {
			t.Errorf("overview is missing %q", want)
		}
	}
	// The overview is the first tier: it has to stay readable at a glance.
	if lines := strings.Count(overview.String(), "\n"); lines > 45 {
		t.Errorf("overview is %d lines; keep the first tier under 45 and push detail into a topic", lines)
	}

	for _, topic := range helpTopics {
		var b bytes.Buffer
		if !printHelp(&b, topic.key) {
			t.Errorf("printHelp(%q) reported an unknown topic", topic.key)
			continue
		}
		if !strings.Contains(b.String(), topic.title) {
			t.Errorf("topic %q page does not show its title", topic.key)
		}
		if strings.Contains(b.String(), "(undefined)") {
			t.Errorf("topic %q page renders an undefined flag", topic.key)
		}
	}

	var all bytes.Buffer
	if !printHelp(&all, "all") {
		t.Fatal(`printHelp("all") reported an unknown topic`)
	}
	flag.CommandLine.SetOutput(os.Stderr)
	if !strings.Contains(all.String(), "-fcidump") {
		t.Error(`printHelp("all") did not dump the flag set`)
	}

	var miss bytes.Buffer
	if printHelp(&miss, "nosuchtopic") {
		t.Error("printHelp reported success for an unknown topic")
	}
	if !strings.Contains(miss.String(), "Help topics:") {
		t.Error("an unknown topic should list the topic index")
	}
}

// TestHelpWrapsToWidth checks that no rendered line runs past helpWidth, which is what
// keeps the pages readable in a narrow terminal. Verbatim body lines are exempt: they
// are pre-formatted examples and tables that must not be re-flowed.
func TestHelpWrapsToWidth(t *testing.T) {
	realFlags(t)
	for _, topic := range helpTopics {
		var b bytes.Buffer
		printHelp(&b, topic.key)
		verbatim := map[string]bool{}
		for _, ln := range topic.body {
			if strings.HasPrefix(ln, "  ") {
				verbatim[ln] = true
			}
		}
		for _, ln := range strings.Split(b.String(), "\n") {
			if verbatim[ln] || len([]rune(ln)) <= helpWidth {
				continue
			}
			t.Errorf("topic %q: line exceeds %d columns:\n%s", topic.key, helpWidth, ln)
		}
	}
	var overview bytes.Buffer
	printHelp(&overview, "")
	for _, ln := range strings.Split(overview.String(), "\n") {
		if len([]rune(ln)) > helpWidth {
			t.Errorf("overview line exceeds %d columns:\n%s", helpWidth, ln)
		}
	}
}
