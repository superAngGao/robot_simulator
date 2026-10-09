/* Independent system-header ABI probe. No handler installation or subreaper SET. */
#define _GNU_SOURCE
#include <stddef.h>
#include <stdio.h>
#include <signal.h>
#include <sys/prctl.h>

int main(void) {
    struct sigaction action;
    int subreaper = -1;
    if (sigaction(SIGCHLD, NULL, &action) != 0 ||
        prctl(PR_GET_CHILD_SUBREAPER, &subreaper, 0, 0, 0) != 0) return 1;
    printf("{\"pointer_size\":%zu,\"ulong_size\":%zu,\"size\":%zu,"
           "\"offsets\":[%zu,%zu,%zu,%zu],\"sigset_size\":%zu,"
           "\"sigchld\":%d,\"no_cldwait\":%d,\"set_subreaper\":%d,"
           "\"get_subreaper\":%d,\"default_handler\":%s,"
           "\"no_cldwait_active\":%s,\"subreaper\":%d}\n",
           sizeof(void *), sizeof(unsigned long), sizeof(struct sigaction),
           offsetof(struct sigaction, sa_handler), offsetof(struct sigaction, sa_mask),
           offsetof(struct sigaction, sa_flags), offsetof(struct sigaction, sa_restorer),
           sizeof(sigset_t), SIGCHLD, SA_NOCLDWAIT, PR_SET_CHILD_SUBREAPER,
           PR_GET_CHILD_SUBREAPER, action.sa_handler == SIG_DFL ? "true" : "false",
           (action.sa_flags & SA_NOCLDWAIT) ? "true" : "false", subreaper);
    return 0;
}
